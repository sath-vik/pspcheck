from torch.utils.data import DataLoader, random_split
from torch.utils import data
from torchvision import transforms
import torch.distributed as dist
import os
import numpy as np
import torch

from utils import (
    RandomHorizontalFlip,
    RandomGaussianBlur,
    RandomEnhance,
    RandomScaleRandomCrop,
    Normalize,
    ToTensor,
    MultiScale,
    Flip,
    FixedScaleCenterCrop
)

class CityscapesDataset(data.Dataset):
    def __init__(self, root_dir='cityscapes_data/data', split='train', 
                 transform=None, random_horizontal_flip=True,
                 random_gaussian_blur=True, random_enhance=True):
        
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split, 'image')
        self.label_dir = os.path.join(root_dir, split, 'label')
        self.filenames = [f for f in os.listdir(self.image_dir) if f.endswith('.npy')]

        # Cityscapes parameters
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        base_size = [256, 256]  # Matches your actual data dimensions
        crop_size = [256, 256]
        scale_range = [0.5, 2.0]
        self.ignore_index = 255  # Keep original ignore index

        if transform is None:
            transform_list = []
            if split == 'train':
                if random_horizontal_flip:
                    transform_list.append(RandomHorizontalFlip())
                if random_gaussian_blur:
                    transform_list.append(RandomGaussianBlur())
                if random_enhance:
                    transform_list.append(RandomEnhance())

                transform_list += [
                    RandomScaleRandomCrop(base_size, crop_size, scale_range, self.ignore_index),
                    Normalize(mean=mean, std=std),
                    ToTensor()
                ]
            else:
                transform_list = [
                    FixedScaleCenterCrop(base_size),
                    Normalize(mean=mean, std=std),
                    ToTensor()
                ]

                if split == 'val':
                    transform_list.append(MultiScale([1.0]))
                    transform_list.append(Flip())

            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.load(os.path.join(self.image_dir, filename))
        label = np.load(os.path.join(self.label_dir, filename)).astype(np.int32)  # Use int32 instead of int64

        # Convert and validate labels before any transformations
        label = np.where(label == 255, -1, label)
        label = np.clip(label, -1, 18)
        
        # Convert to int32 explicitly
        label = label.astype(np.int32)
        
        # Verify dtype and value range before transformations
        if label.dtype != np.int32:
            raise TypeError(f"Label dtype {label.dtype} is not int32 in {filename}")
            
        # Strict validation with enhanced error reporting
        unique_labels, counts = np.unique(label, return_counts=True)
        invalid_mask = (unique_labels < -1) | (unique_labels >= 19)
        
        if np.any(invalid_mask):
            invalid_values = unique_labels[invalid_mask]
            invalid_counts = counts[invalid_mask]
            invalid_total = np.sum(invalid_counts)
            
            print(f"Critical error in {filename}:")
            print(f"Invalid values: {invalid_values.tolist()}")
            print(f"Total invalid pixels: {invalid_total}")
            print("Value distribution:")
            for v, c in zip(unique_labels, counts):
                print(f"Class {v}: {c} pixels")
            
            # Find first invalid coordinate
            first_invalid = np.argwhere(np.isin(label, invalid_values))[0]
            print(f"First invalid coordinate: {first_invalid}")
            
            # Visualize problematic area
            h, w = label.shape
            y_start = max(0, first_invalid[0]-2)
            y_end = min(h, first_invalid[0]+3)
            x_start = max(0, first_invalid[1]-2)
            x_end = min(w, first_invalid[1]+3)
            print("Problematic region:")
            print(label[y_start:y_end, x_start:x_end])
            
            raise ValueError(f"Invalid labels {invalid_values} in {filename}")

        # Apply transformations AFTER validation
        sample = {
            'image': {'original_scale': image},
            'label': {'semantic_logit': label},
            'filename': filename
        }
        
        if self.transform:
            sample = self.transform(sample)
            
            # Additional validation after transformations
            transformed_label = sample['label']['semantic_logit'].numpy()
            unique_transformed = np.unique(transformed_label)
            invalid_transformed = (unique_transformed < -1) | (unique_transformed >= 19)
            
            if np.any(invalid_transformed):
                raise RuntimeError(
                    f"Transformations introduced invalid labels in {filename}: "
                    f"{unique_transformed[invalid_transformed]}"
                )

        return sample

    def __len__(self):
        return len(self.filenames)

def load_data(args):
    full_train_dataset = CityscapesDataset(split='train')
    train_size = int(0.8 * len(full_train_dataset))
    test_size = len(full_train_dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        full_train_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset = CityscapesDataset(split='val')

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * args.device_num if not args.distributed else args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=True
    )

    return train_loader, val_loader, test_loader
