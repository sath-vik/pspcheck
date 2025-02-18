import torch
import numpy as np
import random
import cv2
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

class MultiScale:
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, sample):
        # Maintain dictionary structure for single sample output
        if len(self.scales) == 1:
            scale = self.scales[0]
            image = sample['image']['original_scale']
            label = sample['label']['semantic_logit']
            
            h, w = image.shape[:2]
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            return {
                'image': {'original_scale': cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)},
                'label': {'semantic_logit': cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)},
                'filename': sample['filename']
            }
        
        # Handle multiple scales with list wrapping
        scaled_samples = []
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        for scale in self.scales:
            h, w = image.shape[:2]
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            scaled_samples.append({
                'image': {'original_scale': cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)},
                'label': {'semantic_logit': cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)},
                'filename': sample['filename']
            })
        return scaled_samples

class Flip:
    def __call__(self, sample):
        # Handle both single samples and lists from MultiScale
        if isinstance(sample, list):
            return [self._flip_item(item) for item in sample]
        return self._flip_item(sample)
    
    def _flip_item(self, item):
        image = item['image']['original_scale']
        label = item['label']['semantic_logit']
        
        # Horizontal flip with copy to avoid negative strides
        return {
            'image': {'original_scale': np.fliplr(image).copy()},
            'label': {'semantic_logit': np.fliplr(label).copy()},
            'filename': item['filename']
        }

class FixedScaleCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        # Maintain dictionary structure
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        h, w = image.shape[:2]
        pad_h = max(self.size[0] - h, 0)
        pad_w = max(self.size[1] - w, 0)
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
        
        h, w = image.shape[:2]
        sh = (h - self.size[0]) // 2
        sw = (w - self.size[1]) // 2
        
        return {
            'image': {'original_scale': image[sh:sh+self.size[0], sw:sw+self.size[1]]},
            'label': {'semantic_logit': label[sh:sh+self.size[0], sw:sw+self.size[1]]},
            'filename': sample['filename']
        }

class SegmentationLosses:
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.CrossEntropyLoss
        raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean' if self.batch_average else 'sum'
        )
        if self.cuda:
            criterion = criterion.cuda()
        return criterion(logit, target)

class SemanticSegmentationMetrics:
    def __init__(self, args):
        self.class_num = 19  # Hardcoded for Cityscapes
        self.ignore_mask = 255
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))
        self.eps = 1e-10

    def update(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.class_num)
        label_true = label_true[mask]
        label_pred = label_pred[mask]
        np.add.at(self.confusion_matrix, (label_true, label_pred), 1)

    def reset(self):
        self.confusion_matrix.fill(0)

    def compute_iou(self):
        hist = self.confusion_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        return np.nanmean(iou)

    def compute_pixel_accuracy(self):
        return np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)

    def compute_mean_pixel_accuracy(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.eps)
        return np.nanmean(np.nan_to_num(class_acc, nan=0.0))

    def __call__(self, prediction, label, mode='train'):
        pred = prediction.argmax(1).cpu().numpy().flatten()
        true = label.cpu().numpy().flatten()
        self.update(true, pred)
        
        return {
            'iou': self.compute_iou(),
            'mean_iou': self.compute_iou(),
            'accuracy': self.compute_pixel_accuracy(),
            'mean_pixel_accuracy': self.compute_mean_pixel_accuracy()
        }
class RandomHorizontalFlip:
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']['original_scale']
            label = sample['label']['semantic_logit']
            
            sample['image']['original_scale'] = np.fliplr(image).copy()
            sample['label']['semantic_logit'] = np.fliplr(label).copy()
        return sample

class RandomGaussianBlur:
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']['original_scale']
            sample['image']['original_scale'] = cv2.GaussianBlur(image, (5,5), 0)
        return sample

class RandomEnhance:
    def __init__(self):
        self.factors = [0.5, 0.75, 1.25, 1.5]
    
    def __call__(self, sample):
        factor = random.choice(self.factors)
        image = sample['image']['original_scale']
        sample['image']['original_scale'] = np.clip(image * factor, 0, 255).astype(np.uint8)
        return sample

class RandomScaleRandomCrop:
    def __init__(self, base_size, crop_size, scale_range, ignore_mask=-1):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.ignore_mask = ignore_mask

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = image.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        pad_h = max(self.crop_size[0] - new_h, 0)
        pad_w = max(self.crop_size[1] - new_w, 0)
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=self.ignore_mask)
        
        h, w = label.shape
        sh = random.randint(0, h - self.crop_size[0])
        sw = random.randint(0, w - self.crop_size[1])
        
        return {
            'image': {'original_scale': image[sh:sh+self.crop_size[0], sw:sw+self.crop_size[1]]},
            'label': {'semantic_logit': label[sh:sh+self.crop_size[0], sw:sw+self.crop_size[1]]},
            'filename': sample['filename']
        }

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image = sample['image']['original_scale'].astype(np.float32) / 255.0
        sample['image']['original_scale'] = (image - self.mean) / self.std
        return sample

class ToTensor:
    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        return {
            'image': {'original_scale': torch.from_numpy(image.transpose(2, 0, 1)).float()},
            'label': {'semantic_logit': torch.from_numpy(label).long()},
            'filename': sample['filename']
        }

def decode_semantic_label(label):
    CITYSCAPES_PALETTE = np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
    ], dtype=np.uint8)
    
    output = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id in range(19):
        output[label == class_id] = CITYSCAPES_PALETTE[class_id]
    output[label == -1] = CITYSCAPES_PALETTE[19]  # Handle ignore index
    return output

def calculate_weigths_labels(dataset, dataloader, num_classes):
    z = np.zeros(num_classes)
    for batch in dataloader:
        labels = batch['label']['semantic_logit'].numpy()
        valid = (labels >= 0) & (labels < num_classes)
        counts = np.bincount(labels[valid], minlength=num_classes)
        z += counts
    return torch.from_numpy(1 / np.log(1.02 + z / z.sum())).float()