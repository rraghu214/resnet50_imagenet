"""
AWS EC2 utilities for EBS volume management and ImageNet preparation
"""
import os
import subprocess
import boto3
import time
from datetime import datetime
import requests
import zipfile
import tarfile
from pathlib import Path

class EBSDataManager:
    """Manage EBS volumes and snapshots for ImageNet data"""
    
    def __init__(self, region='us-east-1'):
        self.ec2 = boto3.client('ec2', region_name=region)
        self.region = region
        
    def create_data_preparation_instance(self, key_name, security_group_id):
        """Create a small instance for data preparation"""
        user_data = """#!/bin/bash
# Update system
apt-get update && apt-get upgrade -y

# Install Python and tools
apt-get install -y python3 python3-pip unzip wget curl awscli

# Install PyTorch CPU version for data preprocessing
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install albumentations pillow tqdm

# Create data directory
mkdir -p /data/imagenet
chown ubuntu:ubuntu /data/imagenet

# Mount additional EBS volume (will be attached separately)
# mkfs.ext4 /dev/xvdf
# mkdir -p /data
# mount /dev/xvdf /data
# echo '/dev/xvdf /data ext4 defaults,nofail 0 2' >> /etc/fstab
"""
        
        response = self.ec2.run_instances(
            ImageId='ami-0c7217cdde317cfec',  # Ubuntu 22.04 LTS
            MinCount=1,
            MaxCount=1,
            InstanceType='t3.medium',  # Cheap instance for data prep
            KeyName=key_name,
            SecurityGroupIds=[security_group_id],
            UserData=user_data,
            BlockDeviceMappings=[
                {
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': 20,  # Small root volume
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True
                    }
                }
            ],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'ImageNet-Data-Prep'},
                        {'Key': 'Purpose', 'Value': 'Data-Preparation'}
                    ]
                }
            ]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        print(f"Created data preparation instance: {instance_id}")
        return instance_id
    
    def create_data_volume(self, size_gb=500, volume_type='gp3'):
        """Create EBS volume for ImageNet data"""
        response = self.ec2.create_volume(
            Size=size_gb,
            VolumeType=volume_type,
            Throughput=250 if volume_type == 'gp3' else None,
            Iops=3000 if volume_type == 'gp3' else None,
            TagSpecifications=[
                {
                    'ResourceType': 'volume',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'ImageNet-Data-Volume'},
                        {'Key': 'Purpose', 'Value': 'ML-Training-Data'}
                    ]
                }
            ]
        )
        
        volume_id = response['VolumeId']
        print(f"Created data volume: {volume_id}")
        
        # Wait for volume to be available
        self._wait_for_volume_available(volume_id)
        return volume_id
    
    def attach_volume_to_instance(self, volume_id, instance_id, device='/dev/xvdf'):
        """Attach volume to instance"""
        response = self.ec2.attach_volume(
            VolumeId=volume_id,
            InstanceId=instance_id,
            Device=device
        )
        
        # Wait for attachment
        self._wait_for_volume_attached(volume_id, instance_id)
        print(f"Attached volume {volume_id} to instance {instance_id}")
        return True
    
    def create_snapshot(self, volume_id, description="ImageNet prepared data"):
        """Create snapshot of prepared data volume"""
        response = self.ec2.create_snapshot(
            VolumeId=volume_id,
            Description=description,
            TagSpecifications=[
                {
                    'ResourceType': 'snapshot',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'ImageNet-Prepared-Data'},
                        {'Key': 'Purpose', 'Value': 'ML-Training-Data'},
                        {'Key': 'Created', 'Value': datetime.now().isoformat()}
                    ]
                }
            ]
        )
        
        snapshot_id = response['SnapshotId']
        print(f"Creating snapshot: {snapshot_id}")
        
        # Wait for snapshot completion
        self._wait_for_snapshot_complete(snapshot_id)
        print(f"Snapshot {snapshot_id} created successfully")
        return snapshot_id
    
    def create_volume_from_snapshot(self, snapshot_id, availability_zone):
        """Create new volume from snapshot for training"""
        response = self.ec2.create_volume(
            SnapshotId=snapshot_id,
            VolumeType='gp3',
            Throughput=500,  # High throughput for training
            Iops=4000,       # High IOPS for training
            AvailabilityZone=availability_zone,
            TagSpecifications=[
                {
                    'ResourceType': 'volume',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'ImageNet-Training-Volume'},
                        {'Key': 'Purpose', 'Value': 'ML-Training'}
                    ]
                }
            ]
        )
        
        volume_id = response['VolumeId']
        self._wait_for_volume_available(volume_id)
        print(f"Created training volume from snapshot: {volume_id}")
        return volume_id
    
    def _wait_for_volume_available(self, volume_id):
        """Wait for volume to be available"""
        waiter = self.ec2.get_waiter('volume_available')
        waiter.wait(VolumeIds=[volume_id])
    
    def _wait_for_volume_attached(self, volume_id, instance_id):
        """Wait for volume to be attached"""
        waiter = self.ec2.get_waiter('volume_in_use')
        waiter.wait(VolumeIds=[volume_id])
    
    def _wait_for_snapshot_complete(self, snapshot_id):
        """Wait for snapshot to complete"""
        waiter = self.ec2.get_waiter('snapshot_completed')
        waiter.wait(SnapshotIds=[snapshot_id])

class ImageNetDownloader:
    """Download and prepare ImageNet dataset"""
    
    def __init__(self, data_dir='/data/imagenet'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_imagenet(self, download_url=None):
        """Download ImageNet dataset
        
        Note: You need to register at https://image-net.org/download.php
        and provide your own download URL
        """
        if download_url is None:
            print("Please register at https://image-net.org/download.php")
            print("Then provide the download URL:")
            download_url = input("ImageNet download URL: ")
        
        print("Downloading ImageNet dataset...")
        print("This will take several hours (150GB+ download)")
        
        # Download training data
        train_file = self.data_dir / "ILSVRC2012_img_train.tar"
        if not train_file.exists():
            print("Downloading training data...")
            self._download_file(f"{download_url}/ILSVRC2012_img_train.tar", train_file)
        
        # Download validation data
        val_file = self.data_dir / "ILSVRC2012_img_val.tar"
        if not val_file.exists():
            print("Downloading validation data...")
            self._download_file(f"{download_url}/ILSVRC2012_img_val.tar", val_file)
        
        # Download development kit (for validation labels)
        devkit_file = self.data_dir / "ILSVRC2012_devkit_t12.tar.gz"
        if not devkit_file.exists():
            print("Downloading development kit...")
            self._download_file(f"{download_url}/ILSVRC2012_devkit_t12.tar.gz", devkit_file)
        
        print("Download completed!")
    
    def extract_and_organize_imagenet(self):
        """Extract and organize ImageNet into train/val folders"""
        print("Extracting and organizing ImageNet...")
        
        # Create directories
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Extract training data
        self._extract_training_data(train_dir)
        
        # Extract validation data
        self._extract_validation_data(val_dir)
        
        print("ImageNet organization completed!")
    
    def _download_file(self, url, filename):
        """Download file with progress bar"""
        try:
            import tqdm
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm.tqdm(
                desc=filename.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
                    
        except ImportError:
            # Fallback without progress bar
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
    
    def _extract_training_data(self, train_dir):
        """Extract training data and organize by class"""
        train_tar = self.data_dir / "ILSVRC2012_img_train.tar"
        
        if not train_tar.exists():
            print(f"Training tar file not found: {train_tar}")
            return
        
        print("Extracting training data...")
        
        # Extract main tar file
        with tarfile.open(train_tar, 'r') as tar:
            tar.extractall(self.data_dir / "train_raw")
        
        # Extract individual class tar files
        train_raw_dir = self.data_dir / "train_raw"
        for class_tar in train_raw_dir.glob("n*.tar"):
            class_name = class_tar.stem
            class_dir = train_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            print(f"Extracting {class_name}...")
            with tarfile.open(class_tar, 'r') as tar:
                tar.extractall(class_dir)
        
        # Clean up raw files
        import shutil
        shutil.rmtree(train_raw_dir)
    
    def _extract_validation_data(self, val_dir):
        """Extract validation data and organize by class"""
        val_tar = self.data_dir / "ILSVRC2012_img_val.tar"
        devkit_tar = self.data_dir / "ILSVRC2012_devkit_t12.tar.gz"
        
        if not val_tar.exists() or not devkit_tar.exists():
            print("Validation files not found")
            return
        
        print("Extracting validation data...")
        
        # Extract validation images
        val_raw_dir = self.data_dir / "val_raw"
        with tarfile.open(val_tar, 'r') as tar:
            tar.extractall(val_raw_dir)
        
        # Extract devkit for labels
        devkit_dir = self.data_dir / "devkit"
        with tarfile.open(devkit_tar, 'r:gz') as tar:
            tar.extractall(devkit_dir)
        
        # Read validation labels
        val_labels_file = devkit_dir / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        with open(val_labels_file, 'r') as f:
            val_labels = [int(line.strip()) for line in f]
        
        # Read class names
        meta_file = devkit_dir / "ILSVRC2012_devkit_t12/data/meta.mat"
        # For simplicity, we'll use the synset names directly
        
        # Organize validation images by class
        val_images = sorted(list(val_raw_dir.glob("ILSVRC2012_val_*.JPEG")))
        
        for img_path, label_idx in zip(val_images, val_labels):
            # Convert 1-indexed to 0-indexed and get synset name
            synset_idx = label_idx - 1
            synset_name = f"n{synset_idx:08d}"  # This is simplified
            
            class_dir = val_dir / synset_name
            class_dir.mkdir(exist_ok=True)
            
            # Move image to class directory
            import shutil
            shutil.move(str(img_path), str(class_dir / img_path.name))
        
        # Clean up
        import shutil
        shutil.rmtree(val_raw_dir)
        shutil.rmtree(devkit_dir)

def create_training_instance_with_data(snapshot_id, key_name, security_group_id, 
                                     instance_type='g4dn.2xlarge', spot=True):
    """Create training instance with data volume from snapshot"""
    
    ec2 = boto3.client('ec2')
    
    # Get availability zones for the instance type
    zones_response = ec2.describe_availability_zones()
    zone = zones_response['AvailabilityZones'][0]['ZoneName']
    
    # Create volume from snapshot
    ebs_manager = EBSDataManager()
    volume_id = ebs_manager.create_volume_from_snapshot(snapshot_id, zone)
    
    # User data for training instance
    user_data = f"""#!/bin/bash
# Update and install essentials
apt-get update && apt-get upgrade -y
apt-get install -y python3 python3-pip git htop nvtop screen

# Install CUDA and PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Mount data volume
mkdir -p /data
mount /dev/xvdf /data
echo '/dev/xvdf /data ext4 defaults,nofail 0 2' >> /etc/fstab

# Clone training repository
cd /home/ubuntu
git clone https://github.com/your-repo/resnet50-imagenet.git
cd resnet50-imagenet
pip3 install -r requirements.txt

# Set permissions
chown -R ubuntu:ubuntu /home/ubuntu/resnet50-imagenet
chown -R ubuntu:ubuntu /data

# Create training script
cat << 'EOF' > /home/ubuntu/start_training.sh
#!/bin/bash
cd /home/ubuntu/resnet50-imagenet
screen -S training -dm bash -c "python3 main.py --dataset imagenet1k --platform ec2 --data-dir /data/imagenet --epochs 200 --batch-size 256 --model resnet50d"
echo "Training started in screen session 'training'"
echo "Use 'screen -r training' to attach"
EOF

chmod +x /home/ubuntu/start_training.sh
chown ubuntu:ubuntu /home/ubuntu/start_training.sh
"""
    
    if spot:
        # Create spot instance request
        response = ec2.request_spot_instances(
            SpotPrice='0.50',  # Max price per hour
            InstanceCount=1,
            LaunchSpecification={
                'ImageId': 'ami-0c7217cdde317cfec',  # Ubuntu 22.04 with NVIDIA drivers
                'InstanceType': instance_type,
                'KeyName': key_name,
                'SecurityGroupIds': [security_group_id],
                'UserData': user_data,
                'BlockDeviceMappings': [
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeSize': 50,
                            'VolumeType': 'gp3',
                            'DeleteOnTermination': True
                        }
                    },
                    {
                        'DeviceName': '/dev/xvdf',
                        'Ebs': {
                            'VolumeId': volume_id,
                            'DeleteOnTermination': False
                        }
                    }
                ],
                'Placement': {'AvailabilityZone': zone}
            },
            Type='one-time'
        )
        
        request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        print(f"Created spot instance request: {request_id}")
        return request_id
    
    else:
        # Create regular instance
        response = ec2.run_instances(
            ImageId='ami-0c7217cdde317cfec',
            MinCount=1,
            MaxCount=1,
            InstanceType=instance_type,
            KeyName=key_name,
            SecurityGroupIds=[security_group_id],
            UserData=user_data,
            BlockDeviceMappings=[
                {
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': 50,
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True
                    }
                },
                {
                    'DeviceName': '/dev/xvdf',
                    'Ebs': {
                        'VolumeId': volume_id,
                        'DeleteOnTermination': False
                    }
                }
            ],
            Placement={'AvailabilityZone': zone}
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        print(f"Created training instance: {instance_id}")
        return instance_id

# Cost optimization utilities
def estimate_training_cost(hours=60, instance_type='g4dn.2xlarge', spot=True):
    """Estimate training cost"""
    
    # Pricing (approximate, varies by region and time)
    pricing = {
        'g4dn.2xlarge': {
            'spot': 0.25,      # Average spot price
            'on_demand': 0.752  # On-demand price
        }
    }
    
    instance_cost = hours * pricing[instance_type]['spot' if spot else 'on_demand']
    
    # Storage costs (for 3 days)
    storage_cost = (500 * 0.08 / 30) * 3  # 500GB gp3 for 3 days
    snapshot_cost = 150 * 0.05 / 30       # Snapshot storage for 1 month
    
    total_cost = instance_cost + storage_cost + snapshot_cost
    
    print(f"Estimated Cost Breakdown:")
    print(f"  Instance ({hours}h): ${instance_cost:.2f}")
    print(f"  Storage (3 days): ${storage_cost:.2f}")
    print(f"  Snapshot (1 month): ${snapshot_cost:.2f}")
    print(f"  Total: ${total_cost:.2f}")
    
    return total_cost

if __name__ == "__main__":
    # Example usage
    print("AWS EC2 Training Cost Estimation:")
    estimate_training_cost(hours=60, spot=True)
    print()
    estimate_training_cost(hours=48, spot=True)
