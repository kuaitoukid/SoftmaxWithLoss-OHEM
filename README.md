# SoftmaxWithLoss-OHEM
SoftmaxWithLoss+OHEM
Main idea
1. Choosing those samples with top loss
2. Dont backward loss for ignored samples
API description
1. use_use_hard_mining: if it is false, it is a traditional SoftmaxWithLoss
2. batch_size: how many samples are taken into consideration
3. hard_ratio: the ratio of hard samples (top most samples in loss) of batch_size, if it is zero, it is just the softmax loss function
