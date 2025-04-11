import re
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot metrics from log files.')
parser.add_argument('log_file_paths', type=str, nargs='+', help='Paths to the log files')
args = parser.parse_args()

# Initialize a dictionary to store the metrics for each file
metrics = {}

# Define the regex pattern to extract the metrics for each epoch
pattern = re.compile(r'epoch=(\d+).*?l1loss,\s+SSIM,\s+PSNR,\s+LPIPS,\s+FID-Score\s+([\d.]+),\s+([\d.]+),\s+([\d.]+),\s+([\d.]+),\s+([\d.]+)', re.DOTALL)

# Read each log file and extract the metrics for each epoch
for log_file_path in args.log_file_paths:
    epochs = []
    l1loss = []
    ssim = []
    psnr = []
    lpips = []
    fid_score = []
    
    with open(log_file_path, 'r') as file:
        content = file.read()
        matches = pattern.findall(content)
        for match in matches:
            epochs.append(int(match[0]))
            l1loss.append(float(match[1]))
            ssim.append(float(match[2]))
            psnr.append(float(match[3]))
            lpips.append(float(match[4]))
            fid_score.append(float(match[5]))
    
    metrics[log_file_path] = {
        'epochs': epochs,
        'l1loss': l1loss,
        'ssim': ssim,
        'psnr': psnr,
        'lpips': lpips,
        'fid_score': fid_score
    }

# Plot the metrics
plt.figure(figsize=(12, 8))

for log_file_path, data in metrics.items():
    plt.subplot(2, 3, 1)
    plt.plot(data['epochs'], data['l1loss'], marker='o', label=log_file_path)
    plt.title('L1 Loss↓')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')

    plt.subplot(2, 3, 2)
    plt.plot(data['epochs'], data['ssim'], marker='o', label=log_file_path)
    plt.title('SSIM↑')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    plt.subplot(2, 3, 3)
    plt.plot(data['epochs'], data['psnr'], marker='o', label=log_file_path)
    plt.title('PSNR↑')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')

    plt.subplot(2, 3, 4)
    plt.plot(data['epochs'], data['lpips'], marker='o', label=log_file_path)
    plt.title('LPIPS↓')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')

    plt.subplot(2, 3, 5)
    plt.plot(data['epochs'], data['fid_score'], marker='o', label=log_file_path)
    plt.title('FID Score↓')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')

plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
plt.tight_layout()
plt.savefig('metrics_comparison_plot.png')
plt.show()
