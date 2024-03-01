# Session 6

## Introduction

This assignment is focussed towards grasping Backpropagation and Architectural Basics. The assignment is divided into 2 parts as described below.

## Part 1

This part involves using MS Excel to perform backprop on a arbitrary defined Neural network.

![Alt text](https://github.com/DhrubaAdhikary/ERA_V2/blob/0cdecd83d9689992800f183b6e9d95183629b156/S6/Part1/comparing_lrs.png)

The above screenshot depicts the training stat while varying the learning rates. We see that as we increase the learning rates we reach the saturation point faster.

The directory also contains `Screenshot 1.png` and `Screenshot 2.png` depicting the working of the backprop algorithm.

#### screenshot1
![Alt text](https://github.com/DhrubaAdhikary/ERA_V2/blob/0cdecd83d9689992800f183b6e9d95183629b156/S6/Part1/Screenshot%201.png)

#### screenshot2
![Alt text](https://github.com/DhrubaAdhikary/ERA_V2/blob/0cdecd83d9689992800f183b6e9d95183629b156/S6/Part1/Screenshot%202.png)

## Part 2

### Target
1. Accuracy > 99.4%
2. Number of Parameters < 20k
3. Num Epochs < 20

### Structure

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
            Conv2d-4           [-1, 16, 24, 24]           2,320
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
            Conv2d-7           [-1, 30, 22, 22]           4,350
              ReLU-8           [-1, 30, 22, 22]               0
       BatchNorm2d-9           [-1, 30, 22, 22]              60
           Conv2d-10           [-1, 30, 11, 11]             930
             ReLU-11           [-1, 30, 11, 11]               0
      BatchNorm2d-12           [-1, 30, 11, 11]              60
           Conv2d-13             [-1, 16, 9, 9]           4,336
             ReLU-14             [-1, 16, 9, 9]               0
      BatchNorm2d-15             [-1, 16, 9, 9]              32
           Conv2d-16             [-1, 16, 7, 7]           2,320
             ReLU-17             [-1, 16, 7, 7]               0
      BatchNorm2d-18             [-1, 16, 7, 7]              32
           Conv2d-19             [-1, 32, 5, 5]           4,640
             ReLU-20             [-1, 32, 5, 5]               0
      BatchNorm2d-21             [-1, 32, 5, 5]              64
        AvgPool2d-22             [-1, 32, 1, 1]               0
           Linear-23                   [-1, 10]             330
================================================================
Total params: 19,698
Trainable params: 19,698
Non-trainable params: 0
----------------------------------------------------------------
```

### Performance Curve
![Alt text](image-1.png)

### Training Stats

```
Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 1
Train: Loss=0.0990 Batch_id=117 Accuracy=87.09: 100%|██████████| 118/118 [00:22<00:00,  5.14it/s]
Test set: Average loss: 0.0666, Accuracy: 9835/10000 (98.35%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 2
Train: Loss=0.1232 Batch_id=117 Accuracy=97.87: 100%|██████████| 118/118 [00:22<00:00,  5.20it/s]
Test set: Average loss: 0.0574, Accuracy: 9831/10000 (98.31%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 3
Train: Loss=0.0495 Batch_id=117 Accuracy=98.22: 100%|██████████| 118/118 [00:23<00:00,  5.00it/s]
Test set: Average loss: 0.0354, Accuracy: 9900/10000 (99.00%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 4
Train: Loss=0.0561 Batch_id=117 Accuracy=98.56: 100%|██████████| 118/118 [00:21<00:00,  5.41it/s]
Test set: Average loss: 0.0288, Accuracy: 9913/10000 (99.13%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 5
Train: Loss=0.0212 Batch_id=117 Accuracy=98.77: 100%|██████████| 118/118 [00:21<00:00,  5.43it/s]
Test set: Average loss: 0.0255, Accuracy: 9918/10000 (99.18%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 6
Train: Loss=0.0566 Batch_id=117 Accuracy=98.82: 100%|██████████| 118/118 [00:21<00:00,  5.38it/s]
Test set: Average loss: 0.0242, Accuracy: 9931/10000 (99.31%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 7
Train: Loss=0.0220 Batch_id=117 Accuracy=98.82: 100%|██████████| 118/118 [00:22<00:00,  5.15it/s]
Test set: Average loss: 0.0228, Accuracy: 9931/10000 (99.31%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 8
Train: Loss=0.0885 Batch_id=117 Accuracy=98.95: 100%|██████████| 118/118 [00:22<00:00,  5.23it/s]
Test set: Average loss: 0.0266, Accuracy: 9911/10000 (99.11%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 9
Train: Loss=0.0089 Batch_id=117 Accuracy=99.01: 100%|██████████| 118/118 [00:23<00:00,  5.11it/s]
Test set: Average loss: 0.0223, Accuracy: 9927/10000 (99.27%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch 10
Train: Loss=0.0120 Batch_id=117 Accuracy=99.08: 100%|██████████| 118/118 [00:23<00:00,  5.12it/s]
Test set: Average loss: 0.0212, Accuracy: 9927/10000 (99.27%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 11
Train: Loss=0.0099 Batch_id=117 Accuracy=99.22: 100%|██████████| 118/118 [00:23<00:00,  5.09it/s]
Test set: Average loss: 0.0173, Accuracy: 9946/10000 (99.46%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 12
Train: Loss=0.0216 Batch_id=117 Accuracy=99.35: 100%|██████████| 118/118 [00:22<00:00,  5.15it/s]
Test set: Average loss: 0.0172, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 13
Train: Loss=0.0535 Batch_id=117 Accuracy=99.37: 100%|██████████| 118/118 [00:22<00:00,  5.22it/s]
Test set: Average loss: 0.0165, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 14
Train: Loss=0.0378 Batch_id=117 Accuracy=99.35: 100%|██████████| 118/118 [00:21<00:00,  5.40it/s]
Test set: Average loss: 0.0166, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 15
Train: Loss=0.0679 Batch_id=117 Accuracy=99.40: 100%|██████████| 118/118 [00:21<00:00,  5.42it/s]
Test set: Average loss: 0.0161, Accuracy: 9947/10000 (99.47%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 16
Train: Loss=0.0075 Batch_id=117 Accuracy=99.38: 100%|██████████| 118/118 [00:22<00:00,  5.19it/s]
Test set: Average loss: 0.0164, Accuracy: 9943/10000 (99.43%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 17
Train: Loss=0.0126 Batch_id=117 Accuracy=99.39: 100%|██████████| 118/118 [00:23<00:00,  5.12it/s]
Test set: Average loss: 0.0163, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 18
Train: Loss=0.0030 Batch_id=117 Accuracy=99.44: 100%|██████████| 118/118 [00:23<00:00,  5.07it/s]
Test set: Average loss: 0.0162, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 19
Train: Loss=0.0026 Batch_id=117 Accuracy=99.40: 100%|██████████| 118/118 [00:23<00:00,  5.06it/s]
Test set: Average loss: 0.0163, Accuracy: 9946/10000 (99.46%)

Adjusting learning rate of group 0 to 5.0000e-03.
Epoch 20
Train: Loss=0.0599 Batch_id=117 Accuracy=99.42: 100%|██████████| 118/118 [00:22<00:00,  5.14it/s]
Test set: Average loss: 0.0162, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 5.0000e-04.
```
