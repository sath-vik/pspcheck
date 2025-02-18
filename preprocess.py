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
        base_size = [256, 256]
        crop_size = [256, 256]
        scale_range = [0.5, 2.0]
        self.ignore_index = -1  # Changed to match label processing

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
                # Validation/test transforms
                transform_list = [
                    FixedScaleCenterCrop(base_size),
                    MultiScale([1.0]) if split == 'val' else None,
                    Flip() if split == 'val' else None,
                    Normalize(mean=mean, std=std),
                    ToTensor()
                ]
                # Remove None values and ensure correct order
                transform_list = [t for t in transform_list if t is not None]

            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.load(os.path.join(self.image_dir, filename))
        label = np.load(os.path.join(self.label_dir, filename)).astype(np.int32)

        # Convert and validate labels
        label = np.where(label == 255, -1, label)
        label = np.clip(label, -1, 18)
        label = label.astype(np.int32)

        # Validate before transformations
        unique_labels = np.unique(label)
        if np.any((unique_labels < -1) | (unique_labels >= 19)):
            invalid = unique_labels[(unique_labels < -1) | (unique_labels >= 19)]
            raise ValueError(f"Invalid labels {invalid} in {filename}")

        sample = {
            'image': {'original_scale': image},
            'label': {'semantic_logit': label},
            'filename': filename
        }

        if self.transform:
            sample = self.transform(sample)

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
