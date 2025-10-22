# AWS EC2 Training Setup Guide

## Complete workflow for cost-optimized ImageNet training on AWS EC2

### Prerequisites
1. AWS Account with appropriate permissions
2. AWS CLI configured (`aws configure`)
3. EC2 Key Pair created
4. Security Group allowing SSH (port 22) and HTTPS (port 443)

## Step 1: One-Time Data Preparation (Cost: ~$2-5)

### 1.1 Create Data Preparation Environment
```bash
# Install required packages
pip install boto3 requests tqdm

# Create the data preparation instance and volume
python -c "
from aws_utils import EBSDataManager, ImageNetDownloader
import boto3

# Create EBS manager
ebs = EBSDataManager(region='us-east-1')

# Create data preparation instance (t3.medium ~ $0.04/hour)
instance_id = ebs.create_data_preparation_instance(
    key_name='your-key-pair-name',
    security_group_id='sg-your-security-group'
)

# Create 500GB data volume
volume_id = ebs.create_data_volume(size_gb=500)

# Attach volume to instance
ebs.attach_volume_to_instance(volume_id, instance_id)

print(f'Data prep instance: {instance_id}')
print(f'Data volume: {volume_id}')
"
```

### 1.2 Download and Prepare ImageNet (Offline Process)
```bash
# SSH into the data preparation instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Format and mount the data volume
sudo mkfs.ext4 /dev/xvdf
sudo mkdir -p /data
sudo mount /dev/xvdf /data
sudo chown ubuntu:ubuntu /data

# Download ImageNet (you need to register at image-net.org first)
cd /data
python3 -c "
from aws_utils import ImageNetDownloader
downloader = ImageNetDownloader('/data/imagenet')
downloader.download_imagenet('YOUR_IMAGENET_DOWNLOAD_URL')
downloader.extract_and_organize_imagenet()
"

# Verify the structure
ls /data/imagenet/
# Should show: train/ val/ (with 1000 subdirectories each)

# The data is now ready - total time: ~6-8 hours
```

### 1.3 Create EBS Snapshot
```bash
# Create snapshot of the prepared data (run from your local machine)
python -c "
from aws_utils import EBSDataManager
ebs = EBSDataManager()
snapshot_id = ebs.create_snapshot('vol-your-volume-id', 'ImageNet 1K Prepared Data')
print(f'Snapshot created: {snapshot_id}')
"

# Terminate the data preparation instance (save money)
aws ec2 terminate-instances --instance-ids i-your-instance-id
```

**Result**: You now have a permanent EBS snapshot with prepared ImageNet data
**Cost**: $2-5 for ~8 hours of t3.medium + storage
**Ongoing cost**: $7.50/month for snapshot storage

## Step 2: Launch Training Instance (Cost: $15-25)

### 2.1 Create Training Instance from Snapshot
```bash
# Launch optimized training instance
python -c "
from aws_utils import create_training_instance_with_data
request_id = create_training_instance_with_data(
    snapshot_id='snap-your-snapshot-id',
    key_name='your-key-pair-name',
    security_group_id='sg-your-security-group',
    instance_type='g4dn.2xlarge',
    spot=True  # Use spot instance for cost savings
)
print(f'Training instance request: {request_id}')
"
```

### 2.2 Start Training
```bash
# SSH into training instance (once it's running)
ssh -i your-key.pem ubuntu@<training-instance-ip>

# Data is already mounted at /data/imagenet - no download needed!
ls /data/imagenet/  # Instant access!

# Start training
cd /home/ubuntu/resnet50-imagenet
./start_training.sh

# Or manually:
screen -S training
python3 main.py --dataset imagenet1k --platform ec2 --data-dir /data/imagenet --epochs 200 --batch-size 256 --model resnet50d
# Ctrl+A, D to detach from screen
```

### 2.3 Monitor Training
```bash
# Check progress (from your local machine or on the instance)
python monitor.py

# Or attach to training screen
ssh -i your-key.pem ubuntu@<instance-ip>
screen -r training
```

## Cost Breakdown (Realistic)

### One-Time Setup:
- **Data prep instance**: t3.medium × 8 hours = $0.32
- **EBS volume**: 500GB × 1 day = $1.33
- **Data transfer**: ImageNet download = $0-5 (depending on source)
- **Total one-time**: **$2-7**

### Per Training Run:
- **Spot instance**: g4dn.2xlarge × 60 hours × $0.25 = **$15**
- **EBS volume**: 500GB × 3 days = **$4**
- **Snapshot storage**: Already paid monthly
- **Total per training**: **$19**

### Monthly Ongoing:
- **Snapshot storage**: 150GB × $0.05 = **$7.50/month**

## Advanced Optimizations

### Spot Instance Best Practices:
```bash
# Set intelligent spot bidding
"SpotPrice": "0.35",  # 50% above typical spot price for reliability
"Type": "persistent",  # Restart if interrupted
```

### Performance Optimizations:
- **EBS Optimized**: Enabled by default on g4dn
- **High IOPS**: 4000 IOPS on training volume
- **Instance storage**: Use local NVMe for temporary files
- **Multi-threading**: 8 data loading workers

### Cost Monitoring:
```bash
# Estimate costs before starting
python -c "
from aws_utils import estimate_training_cost
estimate_training_cost(hours=60, spot=True)
"
```

## Troubleshooting

### Common Issues:

1. **Spot Instance Interrupted**:
   - Training automatically resumes from latest checkpoint
   - No data loss - checkpoints saved every 10 epochs

2. **EBS Volume Not Mounting**:
   ```bash
   # Check if volume is attached
   lsblk
   
   # Manual mount if needed
   sudo mount /dev/xvdf /data
   ```

3. **ImageNet Structure Invalid**:
   ```bash
   # Verify structure
   python3 -c "
   from datasets import verify_imagenet_structure
   valid, msg = verify_imagenet_structure('/data/imagenet')
   print(msg)
   "
   ```

## Expected Results

### Training Performance:
- **Duration**: 48-72 hours for 200 epochs
- **Target accuracy**: >78% validation
- **Throughput**: ~500 images/second on g4dn.2xlarge

### Cost Comparison:
| Method | Cost | Time | Reliability |
|--------|------|------|-------------|
| On-demand | $45 | 60h | 100% |
| Spot (this guide) | $19 | 60h | 95% |
| Colab Pro | $10/month | Limited | 80% |

**Conclusion**: EBS snapshot approach gives you reliable, fast, and cost-effective training at **$15-25 total cost**!
