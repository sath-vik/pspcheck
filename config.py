import argparse

def load_config():
    parser = argparse.ArgumentParser('Segmentation')
    
    # Dataset parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_root', type=str, default='./cityscapes_data/data')
    parser.add_argument('--n_classes', type=int, default=19)
    parser.add_argument('--ignore_mask', type=int, default=255)  # Cityscapes uses 255 for ignore
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--max_iteration', type=int, default=2001)
    parser.add_argument('--warmup_iteration', type=int, default=1500)
    parser.add_argument('--intervals', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # System config
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--result_save', action='store_true', default=True)
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser.parse_args()
