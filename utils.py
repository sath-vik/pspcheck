import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = np.array(sample['image']['original_scale']).astype(np.float32)
        image /= 255
        image -= self.mean
        image /= self.std

        sample['image']['original_scale'] = image

        return sample


class RandomGaussianBlur(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = Image.fromarray(sample['image']['original_scale'])
        if np.random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        sample['image']['original_scale'] = np.array(image)

        return sample


class RandomEnhance(object):
    def __init__(self):
        self.enhance_method = [ImageEnhance.Contrast, ImageEnhance.Brightness, ImageEnhance.Sharpness]

    def __call__(self, sample):
        np.random.shuffle(self.enhance_method)
        image = Image.fromarray(sample['image']['original_scale'])

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(image)
                factor = float(1 + np.random.random() / 10)
                image = enhancer.enhance(factor)

        sample['image']['original_scale'] = np.array(image)
        return sample


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']

        if np.random.random() < 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)

        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label

        return sample


class RandomScaleRandomCrop(object):
    def __init__(self, base_size, crop_size, scale_range=(0.5, 2.0), ignore_mask=255):

        if '__iter__' not in dir(base_size):
            self.base_size = (base_size, base_size)
        else:
            self.base_size = base_size

        if '__iter__' not in dir(crop_size):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.scale_range = scale_range
        self.ignore_mask = ignore_mask

    def __call__(self, sample):
        image = Image.fromarray(sample['image']['original_scale'])
        label = Image.fromarray(sample['label']['semantic_logit'].astype(np.int32), mode='I')

        width, height = image.size
        scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]

        if width > height:
            resize_height = int(scale * self.base_size[1])
            resize_width = int(width * (resize_height / height))

        else:
            resize_width = int(scale * self.base_size[0])
            resize_height = int(height * (resize_width / width))

        image = image.resize((resize_width, resize_height), Image.BILINEAR)
        label = label.resize((resize_width, resize_height), Image.NEAREST)

        padding = [0, 0]
        if resize_width < self.crop_size[0]:
            padding[0] = self.crop_size[0] - resize_width

        if resize_height < self.crop_size[1]:
            padding[1] = self.crop_size[1] - resize_height

        if np.sum(padding) != 0:
            image = ImageOps.expand(image, (0, 0, *padding), fill=0)
            label = ImageOps.expand(label, (0, 0, *padding), fill=self.ignore_mask)

        width, height = image.size
        crop_coordinate = np.array([np.random.randint(0, width - self.crop_size[0] + 1),
                                    np.random.randint(0, height - self.crop_size[1] + 1)])

        image = image.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))
        label = label.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))

        sample['image']['original_scale'] = np.array(image)
        sample['label']['semantic_logit'] = np.array(label)

        return sample


class FixedScaleCenterCrop(object):
    def __init__(self, crop_size):
        if '__iter__' not in dir(crop_size):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, sample):
        image = Image.fromarray(sample['image']['original_scale'])
        label = Image.fromarray(sample['label']['semantic_logit'].astype(np.int32), mode='I')

        width, height = image.size

        if width > height:
            resize_height = int(self.crop_size[1])
            resize_width = int(width * (resize_height / height))

        else:
            resize_width = int(self.crop_size[0])
            resize_height = int(height * (resize_width / width))

        image = image.resize((resize_width, resize_height), Image.BILINEAR)
        label = label.resize((resize_width, resize_height), Image.NEAREST)

        crop_coordinate = np.array([int(resize_width - self.crop_size[0]) // 2,
                                    int(resize_height - self.crop_size[1]) // 2])

        image = image.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))
        label = label.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))

        sample['image']['original_scale'] = np.array(image)
        sample['label']['semantic_logit'] = np.array(label)

        return sample


class Resize(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']

        image = F.interpolate(image.expand(1, *image.shape), scale_factor=self.scale, mode='bilinear', align_corners=False)
        label = F.interpolate(label.float().expand(1, 1, *label.shape), scale_factor=self.scale, mode='nearest')

        sample['image']['original_scale'] = image.squeeze(0)
        sample['label']['semantic_logit'] = label.squeeze().long()

        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.array(sample['image']['original_scale']).astype(np.float32)
        label = np.array(sample['label']['semantic_logit'])
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label)

        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label

        return sample


class MultiScale(object):
    def __init__(self, scale_list):
        self.scale_list = scale_list

    def __call__(self, sample):
        image = sample['image']['original_scale']
        images = dict()

        for scale in self.scale_list:
            if scale == 1:
                images['original_scale'] = image
            else:
                images[str(scale)] = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
        sample['image'] = images

        return sample


class Flip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        images = sample['image']

        old_keys = list(images.keys())
        for key in old_keys:
            images.update({key + '_flip': torch.flip(images[key], dims=[-1])})

        sample['image'] = images
        return sample


# Reference
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/calculate_weights.py
def calculate_weigths_labels(dataset, dataloader, num_classes):
    classes_weights_path = os.path.join('weight', '{}_classes_weight_ratios.npy'.format(dataset))
    if os.path.isfile(classes_weights_path):
        return np.load(classes_weights_path)
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader, desc="Processing Labels")
    # tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']['semantic_logit']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    np.save(classes_weights_path, ret)

    return ret


# Reference
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/loss.py
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SemanticSegmentationMetrics:
    """
    A class to compute semantic segmentation metrics such as:
    - Pixel Accuracy
    - Mean Pixel Accuracy
    - IoU per Class
    - Mean IoU (mIoU)
    """

    def __init__(self, FLAGS):
        """
        Initialize the metrics class.

        Args:
            FLAGS: Configuration object with the following attributes:
                - n_classes: Number of classes in the dataset.
                - ignore_mask: Value in the label to ignore during evaluation.
        """
        self.class_num = FLAGS.n_classes  # Number of classes
        self.ignore_mask = FLAGS.ignore_mask  # Ignore mask value
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))  # Confusion matrix

    def __call__(self, prediction, label):
        """
        Update the confusion matrix with predictions and ground truth labels.

        Args:
            prediction: Tensor of model predictions.
            label: Tensor of ground truth labels.
        """
        self.compute_confusion_matrix_and_add_up(label, prediction)

    def clear(self):
        """Reset the confusion matrix."""
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))

    def compute_confusion_matrix(self, label, prediction):
        """
        Compute the confusion matrix for a batch of predictions and labels.

        Args:
            label: Tensor of ground truth labels.
            prediction: Tensor of model predictions.

        Returns:
            Confusion matrix for the current batch.
        """
        # Ensure labels and predictions are in the correct shape
        if len(label.shape) == 4:  # If one-hot encoded
            label = torch.argmax(label, dim=1)
        if len(prediction.shape) == 4:
            prediction = torch.argmax(prediction, dim=1)

        # Flatten tensors and convert to numpy arrays
        label = label.flatten().cpu().numpy().astype(np.int64)
        prediction = prediction.flatten().cpu().numpy().astype(np.int64)

        # Mask out invalid indices (e.g., ignore mask or out-of-range values)
        valid_indices = (label != self.ignore_mask) & (0 <= label) & (label < self.class_num)

        # Map valid indices into a single array for bincounting
        enhanced_label = self.class_num * label[valid_indices].astype(np.int32) + prediction[valid_indices]
        
        # Compute confusion matrix using bincount
        confusion_matrix = np.bincount(enhanced_label, minlength=self.class_num * self.class_num)
        confusion_matrix = np.reshape(confusion_matrix, (self.class_num, self.class_num))

        return confusion_matrix

    def compute_confusion_matrix_and_add_up(self, label, prediction):
        """
        Update the global confusion matrix with a new batch.

        Args:
            label: Ground truth labels.
            prediction: Model predictions.
        """
        self.confusion_matrix += self.compute_confusion_matrix(label, prediction)

    def get_pixel_accuracy(self):
        """
        Compute overall pixel accuracy.

        Returns:
            Pixel accuracy as a float.
        """
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    def get_mean_pixel_accuracy(self):
        """
        Compute mean pixel accuracy across all classes.

        Returns:
            Mean pixel accuracy as a float.
            Ignores classes with no ground truth pixels to avoid `nan`.
        """
        per_class_accuracy = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        
        # Replace nan values with 0 for classes without ground truth pixels
        per_class_accuracy = np.nan_to_num(per_class_accuracy)

        return np.mean(per_class_accuracy)

    def get_iou_per_class(self):
        """
        Compute Intersection over Union (IoU) for each class.

        Returns:
            A numpy array containing IoU for each class.
            Classes without valid pixels will have IoU set to 0.
        """
        iou_per_class = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=0) + 
            np.sum(self.confusion_matrix, axis=1) - 
            np.diag(self.confusion_matrix)
        )
        
        # Replace nan values with 0 for classes without valid pixels
        iou_per_class = np.nan_to_num(iou_per_class)

        return iou_per_class

    def get_mean_iou(self):
        """
        Compute mean Intersection over Union (mIoU).

        Returns:
            Mean IoU as a float. Ignores invalid classes during computation.
        """
        iou_per_class = self.get_iou_per_class()
        
        # Compute mean IoU only for valid classes (non-zero denominators)
        return np.mean(iou_per_class)
