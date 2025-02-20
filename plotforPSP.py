import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the CSV file
    data_file = 'PreviousRuns/training_log.csv'
    df = pd.read_csv(data_file)
    
    # Set plot style
    plt.style.use('ggplot')
    
    # Define metrics and their display names
    metrics = {
        'loss': 'Loss',
        'lr': 'Learning Rate',
        'iou': 'IoU'
    }
    
    window_size = 100  # Moving average window
    
    for metric, label in metrics.items():
        # Calculate moving average
        df[f'{metric}_ma'] = df[metric].rolling(window=window_size, min_periods=1).mean()
        
        # Filter valid data
        metric_data = df.dropna(subset=[metric])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot raw data
        plt.scatter(
            metric_data['epoch'],
            metric_data[metric],
            s=10,
            alpha=0.3,
            color='#1f77b4',
            label='Raw Data'
        )
        
        # Plot moving average
        plt.plot(
            metric_data['epoch'],
            metric_data[f'{metric}_ma'],
            linewidth=2,
            color='#2ca02c',
            label=f'Moving Average ({window_size} epochs)'
        )
        
        # Customize plot
        plt.title(f'Training {label} vs Epoch', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(f'Training {label}', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'{metric}_vs_epoch.png', dpi=300)
        plt.close()
        print(f"Plot saved as {metric}_vs_epoch.png")

if __name__ == "__main__":
    main()
