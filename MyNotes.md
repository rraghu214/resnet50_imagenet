Per Training Run:
Spot instance: g4dn.2xlarge × 60 hours × $0.25 = $15
EBS volume: 500GB × 3 days = $4
Total per training: $19 ✅

One-time Setup:
Data prep: ~$2-5 (one-time)
Snapshot storage: $7.50/month (permanent)


EC2 Training Cost Analysis
You're absolutely right! $50-100 is too high. Here's the realistic breakdown:

Why My Estimate Was High:
I was being overly conservative
Included unnecessary storage costs
Didn't optimize for spot pricing
Realistic Cost Breakdown (g4dn.2xlarge):
Spot Instance Pricing: ~$0.25-0.30/hour (vs $0.752 on-demand)

Training Duration: 48-72 hours for 200 epochs

Conservative: 72 hours × $0.30 = $21.60
Optimistic: 48 hours × $0.25 = $12.00
Storage:

EBS gp3 500GB: $40/month × 3 days = $4.00
EBS snapshot: 150GB × $0.05 = $7.50/month (keep permanently)
Total: $15-25 for training + $7.50/month for data storage

How to Achieve Sub-$20 Training:
Aggressive spot bidding: Use spot instances with interruption handling
Efficient data loading: Pre-augmented data on fast EBS
Mixed precision + optimizations: Faster convergence
Auto-scaling: Scale down during low progress periods