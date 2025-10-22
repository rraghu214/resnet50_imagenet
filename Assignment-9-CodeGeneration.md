1. Give me a simple to understand modular python code base where I can run it on my personal Nvidia machine or Google Colab or EC2.  
2. Modularity should be such a way that (feel free to modify as needed):
  - augmentations are in one class, 
  - optimizers-schedulers in one class, 
  - training in one class, 
  - pbar (drawing progress, generating graph every 10 epochs ) in one class, 
  - the entire orchestration in one class, 
  - the logs generated to be saved in one folder along with graph of every 10 epochs. Every epoch run when printed contains good details as shown in the example below.
    [START 2025-10-10 06:59:17] Epoch 53/100
    Epoch 53: 100% 391/391 [01:10<00:00,  5.54it/s, Loss=1.8894, Acc=65.05%]
    Validating: 100% 79/79 [00:04<00:00, 19.12it/s]
    Epoch 53/100:
      Train Loss: 1.8894, Train Acc: 65.05%
      Val Loss: 1.6333, Val Acc: 72.79%
      Learning Rate: 0.075640
      Best Val Acc: 72.79% (Epoch 53)
    New best model saved with validation accuracy: 72.79%
    [END   2025-10-10 07:00:33] Epoch 53/100
    --------------------------------------------------
  
3. Use resnet 50, keep another option as resnet-d which will just be a standby in case I need to use it. I should also be able to select between imagenet-1K, imagenet-100K, tiny imagenet.
4. For google collab usecase mount the drive to /content/drive and download the image to the data dir - /content/drive/MyDrive/datasets
5. For EC2, suggest a better option where I can download the image apply augmentations and upload to ebs as an offline process so that I don't consume the GPUs training time.


Remember, keep it simple to understand and modular.

Main Instructions
1. You are training ResNet50 from scratch on EC2 on ImageNet
2. Target is to achieve more than 78% accuracy on validation set
3. This needs to be run on AWS EC2 with as low budget as possible. EC2 is with 8vCPUs and G type spot instance of g4dn.2xlarge.
4. I dont want to run on EC2 unless I have tried to run the model on some free resources like my personal machine with Nvidia Geforce RTX 2070 with Max-Q design to verify the code or Google Collab with T4 GPU.
5. Provide a clear strategy on how to accomplish this which includes augmentation strategy, optimizer, scheduler and any other pieces that I must be aware of.

Based on my approval only you should generate pytorch code.