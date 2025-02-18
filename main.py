import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms

from model import PSPNet
from config import load_config
from preprocess import load_data
from utils import decode_semantic_label, SemanticSegmentationMetrics

import os
import numpy as np
import cv2

def save_checkpoint(model, optimizer, scheduler, args, global_step, scope=None):
    if scope is None:
        scope = global_step

    if args.distributed is False or args.local_rank == 0:
        model_state_dict = model.module.state_dict() if args.device_num > 1 else model.state_dict()
        
        torch.save({
            'model_state_dict': model_state_dict,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join('checkpoint', f'checkpoint_model_{scope}.pth'))

class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.warmup_iteration = warmup_iteration
        super().__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        ratio = max(0.0, 1.0 - (step / self.max_iteration))
        return base_lr * (ratio ** self.gamma)

    def warmup_lr(self, base_lr, alpha):
        return base_lr * ((1 - alpha)/10.0 + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            return [
                min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch))
                for base_lr in self.base_lrs
            ]
        return [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

def train(global_step, train_loader, model, optimizer, criterion, scheduler, args):
    model.train()
    metric = SemanticSegmentationMetrics(args)
    
    for data in train_loader:
        if global_step > args.max_iteration:
            break
            
        img = {k: v.cuda() for k, v in data['image'].items()} if args.cuda else data['image']
        label = data['label']['semantic_logit'].cuda() if args.cuda else data['label']['semantic_logit']

        logit, loss = model(img, label.long())
        evaluation = metric(logit, label)
        
        print(f'[Global Step: {global_step}] '
              f'[Loss: {loss.item():.4f}], '
              f'[Acc: {evaluation["accuracy"]:.3f}], '
              f'[mAcc: {evaluation["mean_pixel_accuracy"]:.3f}], '
              f'[mIoU: {evaluation["mean_iou"]:.3f}]')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if global_step % args.intervals == 0:
            save_checkpoint(model, optimizer, scheduler, args, global_step)
            
        global_step += 1
        
    return global_step

def eval(loader, model, args, mode='val'):
    metric = SemanticSegmentationMetrics(args)
    metric.reset()
    model.eval()
    
    with torch.no_grad():
        for idx, data in enumerate(loader):
            img = {k: v.cuda() for k, v in data['image'].items()} if args.cuda else data['image']
            label = data['label']['semantic_logit'].cuda() if args.cuda else data['label']['semantic_logit']

            logit = model(img)
            evaluation = metric(logit, label)
            
            if idx % 10 == 0:
                print(f'[{mode.upper()}][{idx+1}/{len(loader)}] '
                      f'mIoU: {evaluation["mean_iou"]:.3f} '
                      f'Acc: {evaluation["accuracy"]:.3f} '
                      f'mAcc: {evaluation["mean_pixel_accuracy"]:.3f}')

            if args.result_save and mode == 'test':
                os.makedirs('result', exist_ok=True)
                pred = logit.argmax(1).squeeze().cpu().numpy()
                pred_viz = decode_semantic_label(pred)
                
                # Convert .npy filename to .png
                original_name = data['filename'][0]
                base_name = os.path.splitext(original_name)[0]
                output_filename = f"{base_name}.png"
                
                cv2.imwrite(os.path.join('result', output_filename), pred_viz)

    final_metrics = {
        'mIoU': metric.compute_iou(),
        'accuracy': metric.compute_pixel_accuracy(),
        'mean_pixel_accuracy': metric.compute_mean_pixel_accuracy()
    }
    
    print(f'\nFinal {mode.upper()} Metrics:')
    print(f'• Mean IoU: {final_metrics["mIoU"]:.4f}')
    print(f'• Pixel Accuracy: {final_metrics["accuracy"]:.4f}')
    print(f'• Mean Pixel Accuracy: {final_metrics["mean_pixel_accuracy"]:.4f}')
    
    return final_metrics


def main(args):
    train_loader, val_loader, test_loader = load_data(args)
    
    model = PSPNet(n_classes=args.n_classes)
    
    if args.cuda:
        model = model.cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    if args.evaluation:
        checkpoint = torch.load('./checkpoint/checkpoint_model_2000.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("\nEvaluating on validation set:")
        eval(val_loader, model, args, 'val')
        print("\nEvaluating on test set:")
        eval(test_loader, model, args, 'test')
    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=args.ignore_mask,
            reduction='mean'
        ).cuda()

        params = [
            {'params': [p for n, p in model.named_parameters() if 'backbone' in n]},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr * 10}
        ]
        
        optimizer = optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
        
        scheduler = PolyLr(
            optimizer,
            gamma=args.gamma,
            max_iteration=args.max_iteration,
            warmup_iteration=args.warmup_iteration
        )

        global_step = 0
        os.makedirs('checkpoint', exist_ok=True)
        
        while global_step < args.max_iteration:
            global_step = train(global_step, train_loader, model, optimizer, criterion, scheduler, args)

        print("\nFinal Validation Evaluation:")
        eval(val_loader, model, args, 'val')
        print("\nFinal Test Evaluation:")
        eval(test_loader, model, args, 'test')

if __name__ == '__main__':
    args = load_config()
    main(args)
