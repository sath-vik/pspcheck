import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import csv
from datetime import datetime
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

np.seterr(divide='ignore', invalid='ignore')

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
    return correct / target.numel()

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        image = sample['image']['original_scale']
        image = (image - self.mean) / self.std
        sample['image']['original_scale'] = image
        return sample

class ToTensor:
    def __call__(self, sample):
        image = sample['image']['original_scale'].transpose(2, 0, 1)
        label = sample['label']['semantic_logit']
        sample['image']['original_scale'] = torch.from_numpy(image).float()
        sample['label']['semantic_logit'] = torch.from_numpy(label).long()
        return sample

class RandomHorizontalFlip:
    def __call__(self, sample):
        if np.random.random() < 0.5:
            image = np.fliplr(sample['image']['original_scale'])
            label = np.fliplr(sample['label']['semantic_logit'])
            sample['image']['original_scale'] = image
            sample['label']['semantic_logit'] = label
        return sample

class RandomScaleRandomCrop:
    def __init__(self, base_size=2048, crop_size=473, scale_range=(0.5, 2.0)):
        self.crop_size = (crop_size, crop_size)
        self.scale_range = scale_range
        self.base_size = base_size

    def __call__(self, sample):
        img = Image.fromarray(sample['image']['original_scale'])
        lbl = Image.fromarray(sample['label']['semantic_logit'].astype(np.uint8))

        scale = np.random.uniform(*self.scale_range)
        w, h = int(img.width * scale), int(img.height * scale)
        img = img.resize((w, h), Image.BILINEAR)
        lbl = lbl.resize((w, h), Image.NEAREST)

        x = np.random.randint(0, w - self.crop_size[0])
        y = np.random.randint(0, h - self.crop_size[1])
        img = img.crop((x, y, x+self.crop_size[0], y+self.crop_size[1]))
        lbl = lbl.crop((x, y, x+self.crop_size[0], y+self.crop_size[1]))

        sample['image']['original_scale'] = np.array(img)
        sample['label']['semantic_logit'] = np.array(lbl)
        return sample

class FixedScaleCenterCrop:
    def __init__(self, crop_size=473):
        self.crop_size = (crop_size, crop_size)

    def __call__(self, sample):
        img = Image.fromarray(sample['image']['original_scale'])
        lbl = Image.fromarray(sample['label']['semantic_logit'].astype(np.uint8))

        w, h = img.size
        ratio = min(self.crop_size[0]/w, self.crop_size[1]/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        lbl = lbl.resize((new_w, new_h), Image.NEAREST)

        x = (new_w - self.crop_size[0]) // 2
        y = (new_h - self.crop_size[1]) // 2
        img = img.crop((x, y, x+self.crop_size[0], y+self.crop_size[1]))
        lbl = lbl.crop((x, y, x+self.crop_size[0], y+self.crop_size[1]))

        sample['image']['original_scale'] = np.array(img)
        sample['label']['semantic_logit'] = np.array(lbl)
        return sample

def calculate_class_weights(dataloader, num_classes=19, ignore_index=255):
    class_counts = np.zeros(num_classes, dtype=np.float32)
    
    print("Calculating class weights...")
    for batch in tqdm(dataloader):
        labels = batch['label']['semantic_logit'].numpy()
        valid = (labels != ignore_index)
        counts = np.bincount(labels[valid], minlength=num_classes)
        class_counts += counts + 1e-6

    class_weights = 1 / np.log(1.02 + class_counts / class_counts.sum())
    class_weights /= class_weights.min()
    return torch.from_numpy(class_weights).float()

class SegmentationLosses:
    def __init__(self, weight=None, ignore_index=255):
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction='mean'
        )

    def __call__(self, outputs, targets):
        return self.criterion(outputs, targets)

class SemanticSegmentationMetrics:
    def __init__(self, num_classes=19, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds, targets):
        preds = torch.argmax(preds, 1).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        mask = (targets != self.ignore_index)
        preds = preds[mask]
        targets = targets[mask]

        np.add.at(self.confusion_matrix, (targets, preds), 1)

    def reset(self):
        self.confusion_matrix.fill(0)

    def get_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(0) + self.confusion_matrix.sum(1) - intersection
        iou = intersection / (union + 1e-8)
        return iou

    def get_miou(self):
        iou = self.get_iou()
        valid = np.where(iou >= 0)[0]
        return np.mean(iou[valid])
    
    def get_pixel_accuracy(self):
        correct = self.confusion_matrix.trace()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-8)

class CSVTrainLogger:
    def __init__(self, filename='training_log.csv'):
        self.filename = filename
        self._create_header()

    def _create_header(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                f.write('epoch,train_loss,train_miou,val_loss,val_miou,time\n')

    def log(self, epoch, train_loss, train_miou, val_loss=None, val_miou=None):
        timestamp = datetime.now().isoformat()
        with open(self.filename, 'a') as f:
            line = f"{epoch},{train_loss:.4f},{train_miou:.4f},"
            line += f"{val_loss:.4f if val_loss else ''},"
            line += f"{val_miou:.4f if val_miou else ''},{timestamp}\n"
            f.write(line)