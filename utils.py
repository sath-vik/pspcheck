import torch
import numpy as np
import random
import cv2
from torchvision import transforms
import torch.nn.functional as F

class MultiScale:
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        scaled_samples = []
        for scale in self.scales:
            h, w = image.shape[:2]
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scaled_label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            scaled_samples.append({
                'image': {'original_scale': scaled_image},
                'label': {'semantic_logit': scaled_label},
                'filename': sample['filename']
            })
        return scaled_samples

class Flip:
    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        # Horizontal flip
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()
        
        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label
        return sample

class FixedScaleCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        h, w = image.shape[:2]
        if h < self.size[0] or w < self.size[1]:
            # Pad if needed
            pad_h = max(self.size[0] - h, 0)
            pad_w = max(self.size[1] - w, 0)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant')
            
        # Center crop
        h, w = image.shape[:2]
        sh = (h - self.size[0]) // 2
        sw = (w - self.size[1]) // 2
        
        image = image[sh:sh+self.size[0], sw:sw+self.size[1]]
        label = label[sh:sh+self.size[0], sw:sw+self.size[1]]
        
        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label
        return sample


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
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                              size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

class SemanticSegmentationMetrics:
    def __init__(self, args):
        self.class_num = 19  # Changed from 21 to 19
        self.ignore_mask = 255
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))
        self.args = args

    def update(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.class_num)
        label_true = label_true[mask]
        label_pred = label_pred[mask]
        np.add.at(self.confusion_matrix, (label_true, label_pred), 1)

    def reset(self):
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))

    def compute_iou(self):
        hist = self.confusion_matrix
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        iou = np.nanmean(iou)
        return iou

    def compute_pixel_accuracy(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        return acc

    def compute_mean_pixel_accuracy(self):
        class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mean_acc = np.nanmean(class_acc)
        return mean_acc

    def compute_mean_iou(self):
        return self.compute_iou()

    def __call__(self, prediction, label, mode='train'):
        pred = prediction.argmax(1).detach().cpu().numpy()
        true = label.detach().cpu().numpy()
        self.update(true.flatten(), pred.flatten())
        
        metrics = {
            'iou': self.compute_iou(),
            'mean_iou': self.compute_mean_iou(),
            'accuracy': self.compute_pixel_accuracy(),
            'mean_pixel_accuracy': self.compute_mean_pixel_accuracy()
        }
        
        if mode == 'val':
            self.reset()
        return metrics

class RandomHorizontalFlip:
    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        if np.random.random() < 0.5:
            # Numpy array flipping
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label
        return sample

class RandomGaussianBlur:
    def __call__(self, sample):
        image = sample['image']['original_scale']
        if np.random.random() < 0.5:
            # Using OpenCV for blurring numpy arrays
            image = cv2.GaussianBlur(image, (5,5), 0)
        sample['image']['original_scale'] = image
        return sample

class RandomEnhance:
    def __init__(self):
        self.factors = [0.5, 0.75, 1.25, 1.5]
    
    def __call__(self, sample):
        image = sample['image']['original_scale']
        factor = random.choice(self.factors)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        sample['image']['original_scale'] = image
        return sample

class RandomScaleRandomCrop:
    def __init__(self, base_size, crop_size, scale_range, ignore_mask=-1):  # Changed ignore_mask to -1
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.ignore_mask = ignore_mask  # Now using -1 for ignore index

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        # Scale the image and label
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = image.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize with proper interpolation
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Calculate padding
        pad_h = max(self.crop_size[0] - new_h, 0)
        pad_w = max(self.crop_size[1] - new_w, 0)
        
        # Pad with ignore index
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=self.ignore_mask)
        
        # Random crop
        h, w = label.shape
        sh = random.randint(0, h - self.crop_size[0])
        sw = random.randint(0, w - self.crop_size[1])
        
        image = image[sh:sh+self.crop_size[0], sw:sw+self.crop_size[1]]
        label = label[sh:sh+self.crop_size[0], sw:sw+self.crop_size[1]]
        
        # Clip labels to valid range
        label = np.clip(label, -1, 18)
        
        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label
        return sample


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image = sample['image']['original_scale'].astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        sample['image']['original_scale'] = image
        return sample

class ToTensor:
    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        # Convert numpy arrays to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()
        
        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label
        return sample

# def decode_semantic_label(label):
#     # Placeholder - replace with Cityscapes color mapping
#     return np.stack([label]*3, axis=-1).astype(np.uint8) * 50

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Preserve original logic but adapt for 19 classes
    z = np.zeros((num_classes,))
    for sample in dataloader:
        y = sample['label']['semantic_logit'].numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    return ret



def decode_semantic_label(label):
    # Cityscapes color palette (19 classes + void)
    CITYSCAPES_PALETTE = np.array([
        [128, 64, 128],       # road
        [244, 35, 232],       # sidewalk
        [70, 70, 70],         # building
        [102, 102, 156],      # wall
        [190, 153, 153],      # fence
        [153, 153, 153],      # pole
        [250, 170, 30],       # traffic light
        [220, 220, 0],        # traffic sign
        [107, 142, 35],       # vegetation
        [152, 251, 152],      # terrain
        [70, 130, 180],       # sky
        [220, 20, 60],        # person
        [255, 0, 0],          # rider
        [0, 0, 142],          # car
        [0, 0, 70],           # truck
        [0, 60, 100],         # bus
        [0, 80, 100],         # train
        [0, 0, 230],          # motorcycle
        [119, 11, 32],        # bicycle
        [0, 0, 0]             # void/ignore
    ], dtype=np.uint8)

    # Convert label indices to RGB colors
    output = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_id in range(19):
        output[label == class_id] = CITYSCAPES_PALETTE[class_id]
    output[label == 255] = CITYSCAPES_PALETTE[19]  # Handle ignore class
    
    return output
