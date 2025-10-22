1. You are training ResNet50 from scratch on EC2 on ImageNet
2. Target is to achieve more than 78% accuracy on validation set
3. This needs to be run on AWS EC2 with as low budget as possible. EC2 is with 8vCPUs and G type spot instance of g4dn.2xlarge.
4. I dont want to run on EC2 unless I have tried to run the model on some free resources like my personal machine with Nvidia Geforce RTX 2070 with Max-Q design to verify the code or Google Collab with T4 GPU.
5. Provide a clear strategy on how to accomplish this which includes augmentation strategy, optimizer, scheduler and any other pieces that I must be aware of.

Based on my approval only you should generate pytorch code.