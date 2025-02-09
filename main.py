import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from model import PSPNet
from config import load_config
from preprocess import load_data
from dataset.prepare_voc import decode_semantic_label
from utils import SegmentationLosses, calculate_weigths_labels, SemanticSegmentationMetrics

import os
import numpy as np
import cv2


def save_checkpoint(model, optimizer, scheduler, args, global_step, scope=None):
    if scope is None:
        scope = global_step
    if not args.distributed or args.local_rank == 0:
        if args.device_num > 1:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        torch.save({
            'model_state_dict': model_state_dict,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join('checkpoint', 'checkpoint_model_{}.pth'.format(scope)))


class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.warmup_iteration = warmup_iteration
        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return base_lr * ((1 - (step / self.max_iteration)) ** (self.gamma))

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch))
                   for base_lr in self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]
        return lrs


def train(global_step, train_loader, model, optimizer, criterion, scheduler, args):
    model.train()
    metric = SemanticSegmentationMetrics(args)
    for data in train_loader:
        if global_step > args.max_iteration:
            break
        print('[Global Step: {0}]'.format(global_step), end=' ')
        img, label = data['image'], data['label']['semantic_logit']

        if args.cuda:
            for key in img.keys():
                img[key] = img[key].cuda()
            label = label.cuda()

        # logit, loss = model(img, label.long())
        # metric(logit, label)  # Update confusion matrix
        # accuracy = metric.get_pixel_accuracy()
        # # evaluation = metric(logit, label)
        # print('[Loss: {0:.4f}], [Accuracy: {:.5f}]'.format(loss.item(), accuracy))
        logit, loss = model(img, label.long())  # Assumes model returns both logits and loss

        # Update confusion matrix
        metric(logit.argmax(dim=1), label)

        # Compute metrics after updating confusion matrix
        accuracy = metric.get_pixel_accuracy()  # Overall pixel accuracy
        
        print(f'[Loss: {loss.item():.4f}], [Accuracy: {accuracy:.5f}]')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if global_step % args.intervals == 0:
            save_checkpoint(model, optimizer, scheduler, args, global_step)
        global_step += 1
    return global_step


def eval(val_loader, model, args):
    metric = SemanticSegmentationMetrics(args)
    metric.clear()
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            img, label = data['image'], data['label']['semantic_logit']
            if args.cuda:
                for key in img.keys():
                    img[key] = img[key].cuda()
                label = label.cuda()

            logit = model(img)
            metric(logit.argmax(dim=1), label)

            # Save predictions if enabled
            if args.result_save:
                if not os.path.isdir('result'):
                    os.mkdir('result')
                pred = logit.argmax(dim=1).squeeze().detach().cpu().numpy()
                pred = decode_semantic_label(pred)
                filename = data['filename'][0]
                if not filename.endswith('.png'):
                    filename += '.png'
                cv2.imwrite(f'result/{filename}', pred.astype(np.uint8))

        # Compute final metrics after all validation samples are processed
        iou_per_class = metric.get_iou_per_class()  # IoU for each class
        mean_iou = metric.get_mean_iou()  # Mean IoU (mIoU)
        pixel_acc = metric.get_pixel_accuracy()  # Overall Pixel Accuracy
        mean_pixel_acc = metric.get_mean_pixel_accuracy()  # Mean Pixel Accuracy

        # Print metrics summary
        print("\nEvaluation Metrics:")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
        print(f"Mean IoU (mIoU): {mean_iou:.4f}")
        
        print("Class-wise IoU:")
        for i, iou in enumerate(iou_per_class):
            print(f"  Class {i}: IoU = {iou:.4f}")


def main(args):
    train_loader, val_loader = load_data(args)
    if args.cuda:
        pass

    # Initialize the PSPNet model with the correct number of classes.
    model = PSPNet(n_classes=args.n_classes)
    if args.cuda:
        model = model.cuda()
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            nn.SyncBatchNorm.convert_sync_batchnorm(model),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    if args.evaluation:
        checkpoint_path = './checkpoint/checkpoint_model_6000.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        eval(val_loader, model, args)
    else:
        class_weights = calculate_weigths_labels(args.dataset, train_loader, args.n_classes)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).cuda(),  # Apply class weights
            ignore_index=args.ignore_mask  # Ignore invalid labels
        )

        backbone_params = nn.ParameterList()
        decoder_params = nn.ParameterList()

        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        params_list = [{'params': backbone_params},
                       {'params': decoder_params, 'lr': args.lr * 10}]

        optimizer = optim.SGD(params_list,
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)

        scheduler = PolyLr(optimizer,
                           gamma=args.gamma,
                           max_iteration=args.max_iteration,
                           warmup_iteration=args.warmup_iteration)

        global_step = 0
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        while global_step < args.max_iteration:
            global_step = train(global_step,
                                train_loader,
                                model,
                                optimizer,
                                criterion,
                                scheduler,
                                args)

if __name__ == '__main__':
    args = load_config()
    main(args)
