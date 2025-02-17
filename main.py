import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from model import PSPNet
from config import load_config
from preprocess import load_data
from utils import SegmentationLosses, calculate_class_weights, SemanticSegmentationMetrics, CSVTrainLogger
import os
import numpy as np
import cv2

def save_checkpoint(model, optimizer, scheduler, args, global_step, scope=None):
    if scope is None:
        scope = global_step
    if not args.distributed or args.local_rank == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join('checkpoint', f'checkpoint_{scope}.pth'))

class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, warmup_iteration=0):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.warmup_iteration = warmup_iteration
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            return [base_lr * (0.1 * (1 - alpha) + alpha) for base_lr in self.base_lrs]
        return [base_lr * ((1 - (self.last_epoch / self.max_iteration)) ** self.gamma) for base_lr in self.base_lrs]

def train(global_step, train_loader, model, optimizer, criterion, scheduler, args, logger):
    model.train()
    metric = SemanticSegmentationMetrics(num_classes=args.n_classes, ignore_index=255)
    total_loss = 0.0
    total_samples = 0
    
    for data in train_loader:
        if global_step > args.max_iteration:
            break
            
        img = data['image']['original_scale']
        label = data['label']['semantic_logit']
        batch_size = img.size(0)
        
        if args.cuda:
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model({'original_scale': img})
        loss = criterion(outputs, label.long())
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric.update(outputs, label)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        current_lr = scheduler.get_last_lr()[0]
        if global_step % 10 == 0:
            print(f'[Step {global_step}] Loss: {loss.item():.4f} | LR: {current_lr:.6f}')
            
        if global_step % 1000 == 0:
            save_checkpoint(model, optimizer, scheduler, args, global_step)
            
        global_step += 1

    return global_step, total_loss/total_samples, metric.get_miou(), scheduler.get_last_lr()[0]

def evaluate(model, loader, args):
    metric = SemanticSegmentationMetrics(num_classes=args.n_classes, ignore_index=255)
    model.eval()
    
    with torch.no_grad():
        for data in loader:
            img = data['image']['original_scale']
            label = data['label']['semantic_logit']
            
            if args.cuda:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            outputs = model({'original_scale': img})
            metric.update(outputs, label)

            if args.result_save:
                pred = outputs.argmax(1).squeeze().cpu().numpy()
                filename = data['filename'][0].replace('/', '_')
                cv2.imwrite(f"result/{filename}.png", pred.astype(np.uint8))

    return {
        'mean_iou': metric.get_miou(),
        'class_iou': metric.get_iou(),
        'pixel_acc': metric.get_pixel_accuracy()
    }

def main():
    args = load_config()
    
    # Setup directories
    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    if args.result_save:
        os.makedirs('result', exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = load_data(args)

    # Initialize model
    model = PSPNet(n_classes=args.n_classes)
    if args.cuda:
        model = model.cuda()

    # Class weights and loss
    class_weights = calculate_class_weights(train_loader, args.n_classes)
    criterion = SegmentationLosses(
        weight=class_weights.cuda() if args.cuda else class_weights,
        ignore_index=255
    )
    
    # Optimizer and scheduler
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': model.decoder.parameters(), 'lr': args.lr * 10}
    ], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    
    scheduler = PolyLr(optimizer, args.gamma, args.max_iteration, args.warmup_iteration)
    logger = CSVTrainLogger('logs/training_log.csv')

    # Training loop
    best_iou = 0.0
    global_step = 0
    try:
        while global_step < args.max_iteration:
            global_step, train_loss, train_iou, lr = train(
                global_step, train_loader, model, optimizer, criterion, scheduler, args, logger
            )

            if global_step % args.intervals == 0:
                val_metrics = evaluate(model, val_loader, args)
                print(f"\nValidation @ {global_step}:")
                print(f"mIoU: {val_metrics['mean_iou']:.4f}")
                
                if val_metrics['mean_iou'] > best_iou:
                    best_iou = val_metrics['mean_iou']
                    save_checkpoint(model, optimizer, scheduler, args, global_step, 'best')

    except KeyboardInterrupt:
        print("\nTraining interrupted!")

    # Final evaluation
    print("\n=== Loading best model ===")
    checkpoint = torch.load('checkpoint/checkpoint_final.pth.pth', 
                          map_location='cuda' if args.cuda else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, args)
    
    print("\n=== Test Results ===")
    print(f"Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"Pixel Accuracy: {test_metrics['pixel_acc']:.4f}")
    print("\nClass-wise IoU:")
    for idx, iou in enumerate(test_metrics['class_iou']):
        print(f"Class {idx:02d}: {iou:.4f}")

    save_checkpoint(model, optimizer, scheduler, args, global_step, 'final')

if __name__ == '__main__':
    main()
