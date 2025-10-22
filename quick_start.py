"""
Quick start scripts for different platforms
"""

# Local machine quick start (Windows PowerShell)
LOCAL_START_PS = """
# Quick start for local testing
Write-Host "Setting up ResNet50 ImageNet training on local machine..." -ForegroundColor Green

# Install dependencies
pip install -r requirements.txt

# Test with ImageNet-100 subset
python main.py --dataset imagenet100 --platform local --epochs 50 --batch-size 128 --lr 1e-3

Write-Host "Training started! Check logs/ directory for progress." -ForegroundColor Green
"""

# Local machine quick start (Linux/Mac)
LOCAL_START_BASH = """#!/bin/bash
# Quick start for local testing
echo "Setting up ResNet50 ImageNet training on local machine..."

# Install dependencies
pip install -r requirements.txt

# Test with ImageNet-100 subset
python main.py --dataset imagenet100 --platform local --epochs 50 --batch-size 128 --lr 1e-3

echo "Training started! Check logs/ directory for progress."
"""

# Google Colab setup
COLAB_SETUP = """
# Google Colab Setup and Training
# Run these cells in Google Colab

# Cell 1: Mount Drive and Setup
from google.colab import drive
drive.mount('/content/drive')

# Clone repository (if not already done)
import os
if not os.path.exists('/content/resnet50-imagenet'):
    !git clone https://your-repo-url.git /content/resnet50-imagenet

%cd /content/resnet50-imagenet

# Cell 2: Install Dependencies
!pip install -r requirements.txt

# Cell 3: Start Training
!python main.py --dataset tiny_imagenet --platform colab --epochs 100 --batch-size 256 --lr 1e-3

# Cell 4: Monitor Progress (run periodically)
import matplotlib.pyplot as plt
from IPython.display import Image, display
import glob

# Show latest training curve
latest_plot = sorted(glob.glob('/content/drive/MyDrive/logs/*/training_curves_latest.png'))
if latest_plot:
    display(Image(latest_plot[-1]))
"""

# AWS EC2 setup script
EC2_SETUP = """#!/bin/bash
# AWS EC2 Setup Script for g4dn.2xlarge
# Run this on your EC2 instance

echo "Setting up ResNet50 ImageNet training on AWS EC2..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Install CUDA and PyTorch (GPU version)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Clone repository
git clone https://your-repo-url.git
cd resnet50-imagenet

# Install dependencies
pip3 install -r requirements.txt

# Configure AWS CLI (you'll need to enter your credentials)
aws configure

# Download ImageNet data (you'll need to provide the download link)
# wget -O imagenet.tar "your-imagenet-download-link"
# tar -xf imagenet.tar -C ./data/

# Start training with screen (so it continues if SSH disconnects)
screen -S resnet_training -dm bash -c "python3 main.py --dataset imagenet1k --platform ec2 --epochs 200 --batch-size 256 --model resnet50d"

echo "Training started in screen session 'resnet_training'"
echo "Use 'screen -r resnet_training' to attach to the session"
echo "Use Ctrl+A, D to detach from screen session"
"""

# Training monitoring script
MONITOR_SCRIPT = """#!/usr/bin/env python3
# Training Monitor Script
# Run this to monitor training progress

import os
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def find_latest_log_dir():
    \"\"\"Find the most recent log directory\"\"\"
    log_dirs = glob.glob('./logs/run_*')
    if not log_dirs:
        return None
    return max(log_dirs, key=os.path.getctime)

def read_metrics(log_dir):
    \"\"\"Read metrics from JSON file\"\"\"
    metrics_file = os.path.join(log_dir, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def print_training_status():
    \"\"\"Print current training status\"\"\"
    log_dir = find_latest_log_dir()
    if not log_dir:
        print("No training logs found.")
        return
    
    metrics = read_metrics(log_dir)
    if not metrics or not metrics.get('epochs'):
        print("No training metrics found.")
        return
    
    # Latest metrics
    latest_epoch = metrics['epochs'][-1]
    latest_train_acc = metrics['train_acc'][-1]
    latest_val_acc = metrics['val_acc'][-1]
    best_val_acc = max(metrics['val_acc'])
    
    print(f"\\n{'='*50}")
    print(f"Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    print(f"Log Directory: {log_dir}")
    print(f"Current Epoch: {latest_epoch}")
    print(f"Latest Train Acc: {latest_train_acc:.2f}%")
    print(f"Latest Val Acc: {latest_val_acc:.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"Target (78%): {'✓ ACHIEVED' if best_val_acc >= 78.0 else f'Need {78.0 - best_val_acc:.2f}% more'}")
    print(f"{'='*50}")

def monitor_training(refresh_interval=60):
    \"\"\"Monitor training with periodic updates\"\"\"
    print("Starting training monitor...")
    print(f"Refreshing every {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print_training_status()
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\\nMonitoring stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_training()
    else:
        print_training_status()
"""

# Create quick start files
def create_quick_start_files():
    """Create platform-specific quick start files"""
    
    # Local PowerShell script
    with open('start_local.ps1', 'w') as f:
        f.write(LOCAL_START_PS)
    
    # Local Bash script  
    with open('start_local.sh', 'w') as f:
        f.write(LOCAL_START_BASH)
    
    # Colab notebook content
    with open('colab_setup.py', 'w') as f:
        f.write(COLAB_SETUP)
    
    # EC2 setup script
    with open('setup_ec2.sh', 'w') as f:
        f.write(EC2_SETUP)
    
    # Monitor script
    with open('monitor.py', 'w') as f:
        f.write(MONITOR_SCRIPT)
    
    print("Quick start files created:")
    print("  - start_local.ps1 (Windows PowerShell)")
    print("  - start_local.sh (Linux/Mac)")
    print("  - colab_setup.py (Google Colab)")
    print("  - setup_ec2.sh (AWS EC2)")
    print("  - monitor.py (Training monitor)")

if __name__ == "__main__":
    create_quick_start_files()
