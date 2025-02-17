import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2

class CityscapesDataset(Dataset):
    def __init__(self, args=None, transform=None, split='train'):
        self.base_dir = args.data_root if args else './cityscapes_data/data'
        self.split = split
        self.image_dir = os.path.join(self.base_dir, split, 'image')
        self.label_dir = os.path.join(self.base_dir, split, 'label')
        self.file_list = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

        # Updated class mapping (19 classes + ignore)
        self.class_mapping = {
            0: 0,    # road
            1: 1,    # sidewalk
            2: 2,    # building
            3: 3,    # wall
            4: 4,    # fence
            5: 5,    # pole
            6: 6,    # traffic light
            7: 7,    # traffic sign
            8: 8,    # vegetation
            9: 9,    # terrain
            10: 10,  # sky
            11: 11,  # person
            12: 12,  # rider
            13: 13,  # car
            14: 14,  # truck
            15: 15,  # bus
            16: 16,  # motorcycle
            17: 17,  # bicycle
            255: 255 # ignore
        }
        # Ensure all possible values are mapped
        self.class_mapping = defaultdict(lambda: 255, self.class_mapping)
        self.transform = transform or transforms.Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor()
        ])
        # unique_vals = set()
        # for mask_file in self.mask_files:
        #     mask = np.load(os.path.join(self.label_dir, mask_file))
        #     unique_vals.update(np.unique(mask))
        # print("Unique mask values:", unique_vals)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        
        # Load and preprocess image
        img = np.load(os.path.join(self.image_dir, fname))
        img = cv2.resize(img, (473, 473), interpolation=cv2.INTER_LINEAR)
        
        # Load and process mask
        mask = np.load(os.path.join(self.label_dir, fname))
        mask = cv2.resize(mask, (473, 473), interpolation=cv2.INTER_NEAREST)
        
        # Remap labels using class mapping
        remapped = np.vectorize(lambda x: self.class_mapping[x])(mask).astype(np.uint8)
        
        sample = {
            'image': {'original_scale': img},
            'label': {'semantic_logit': remapped},
            'filename': fname
        }
        return self.transform(sample)

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        img = sample['image']['original_scale']
        img = (img - self.mean) / self.std
        sample['image']['original_scale'] = img
        return sample

class ToTensor:
    def __call__(self, sample):
        img = sample['image']['original_scale']
        label = sample['label']['semantic_logit']
        
        # Add channel dimension for images (assuming input is HxWxC)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            
        sample['image']['original_scale'] = torch.from_numpy(img.transpose(2, 0, 1)).float()
        sample['label']['semantic_logit'] = torch.from_numpy(label).long()
        return sample

def load_data(args):
    # Load full training dataset
    full_train = CityscapesDataset(args, split='train')
    
    # Split into 80% train, 20% test
    train_size = int(0.8 * len(full_train))
    test_size = len(full_train) - train_size
    train_dataset, test_dataset = random_split(full_train, [train_size, test_size])
    
    # Load validation dataset
    val_dataset = CityscapesDataset(args, split='val')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
