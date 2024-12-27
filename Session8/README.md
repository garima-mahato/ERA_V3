# Session 8 - Advanced Neural Network Architectures

## Assignment

|Name|Code Link |
|---|---|
|Execution Jupyter Notebook|[Open](https://github.com/garima-mahato/ERA_V3/blob/main/Session8/S8_Assignment.ipynb)|
| Data Loader Code | [Open](https://github.com/garima-mahato/ERA_V3/blob/main/gmo/data_engine/data_loader.py) |
| Data Augmenter Code | [Open](https://github.com/garima-mahato/ERA_V3/blob/main/gmo/data_engine/data_augmenter.py) |
| Model Code | [Open](https://github.com/garima-mahato/ERA_V3/blob/main/gmo/model.py) |
| Utility Code | [Open](https://github.com/garima-mahato/ERA_V3/blob/main/gmo/utils.py) |

#### Results: 
  - Best Train Accuracy - 87.9%
  - Best Test Accuracy - 85.66%
  - Test Accuracy - 85.51%
  - Total Parameters - 198,660
  - Number of Epochs - 20

### Model Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 64, 32, 32]          18,432
             ReLU-10           [-1, 64, 32, 32]               0
      BatchNorm2d-11           [-1, 64, 32, 32]             128
          Dropout-12           [-1, 64, 32, 32]               0
           Conv2d-13           [-1, 32, 32, 32]           2,048
           Conv2d-14           [-1, 32, 28, 28]           9,216
             ReLU-15           [-1, 32, 28, 28]               0
      BatchNorm2d-16           [-1, 32, 28, 28]              64
          Dropout-17           [-1, 32, 28, 28]               0
           Conv2d-18           [-1, 62, 28, 28]          17,856
             ReLU-19           [-1, 62, 28, 28]               0
      BatchNorm2d-20           [-1, 62, 28, 28]             124
          Dropout-21           [-1, 62, 28, 28]               0
           Conv2d-22           [-1, 72, 28, 28]          40,176
             ReLU-23           [-1, 72, 28, 28]               0
      BatchNorm2d-24           [-1, 72, 28, 28]             144
          Dropout-25           [-1, 72, 28, 28]               0
           Conv2d-26           [-1, 32, 28, 28]           2,304
           Conv2d-27           [-1, 32, 24, 24]           9,216
             ReLU-28           [-1, 32, 24, 24]               0
      BatchNorm2d-29           [-1, 32, 24, 24]              64
          Dropout-30           [-1, 32, 24, 24]               0
           Conv2d-31           [-1, 64, 24, 24]          18,432
             ReLU-32           [-1, 64, 24, 24]               0
      BatchNorm2d-33           [-1, 64, 24, 24]             128
          Dropout-34           [-1, 64, 24, 24]               0
           Conv2d-35           [-1, 64, 26, 26]              64
           Conv2d-36           [-1, 84, 26, 26]           5,376
  SeparableConv2d-37           [-1, 84, 26, 26]               0
             ReLU-38           [-1, 84, 26, 26]               0
      BatchNorm2d-39           [-1, 84, 26, 26]             168
          Dropout-40           [-1, 84, 26, 26]               0
           Conv2d-41           [-1, 84, 22, 22]          63,504
             ReLU-42           [-1, 84, 22, 22]               0
      BatchNorm2d-43           [-1, 84, 22, 22]             168
          Dropout-44           [-1, 84, 22, 22]               0
        AvgPool2d-45             [-1, 84, 1, 1]               0
           Conv2d-46             [-1, 10, 1, 1]             840
================================================================
Total params: 198,660
Trainable params: 198,660
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.84
Params size (MB): 0.76
Estimated Total Size (MB): 14.61
----------------------------------------------------------------

```

![](https://raw.githubusercontent.com/garima-mahato/ERA_V3/main/Session9/assets/cifar10_s8__finaltorchviz.png)

---


##### <b>Train/Test Logs</b>

```
Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 1
  0%|          | 0/391 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
Train: Loss=1.0656 Batch_id=390 Accuracy=42.70: 100%|██████████| 391/391 [00:30<00:00, 12.75it/s]
Test set: Average loss: 1.3007, Accuracy: 5258/10000 (52.58%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 2
Train: Loss=1.0857 Batch_id=390 Accuracy=59.41: 100%|██████████| 391/391 [00:30<00:00, 12.85it/s]
Test set: Average loss: 1.0396, Accuracy: 6290/10000 (62.90%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 3
Train: Loss=0.9856 Batch_id=390 Accuracy=66.17: 100%|██████████| 391/391 [00:32<00:00, 12.19it/s]
Test set: Average loss: 0.9272, Accuracy: 6745/10000 (67.45%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 4
Train: Loss=0.7146 Batch_id=390 Accuracy=70.75: 100%|██████████| 391/391 [00:30<00:00, 12.79it/s]
Test set: Average loss: 0.7617, Accuracy: 7333/10000 (73.33%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 5
Train: Loss=0.8645 Batch_id=390 Accuracy=73.59: 100%|██████████| 391/391 [00:30<00:00, 12.67it/s]
Test set: Average loss: 0.8671, Accuracy: 7062/10000 (70.62%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 6
Train: Loss=0.7622 Batch_id=390 Accuracy=75.55: 100%|██████████| 391/391 [00:30<00:00, 12.90it/s]
Test set: Average loss: 0.8155, Accuracy: 7244/10000 (72.44%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 7
Train: Loss=0.6482 Batch_id=390 Accuracy=77.43: 100%|██████████| 391/391 [00:30<00:00, 12.72it/s]
Test set: Average loss: 0.6510, Accuracy: 7802/10000 (78.02%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 8
Train: Loss=0.6252 Batch_id=390 Accuracy=78.52: 100%|██████████| 391/391 [00:30<00:00, 13.00it/s]
Test set: Average loss: 0.6061, Accuracy: 7921/10000 (79.21%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 9
Train: Loss=0.6144 Batch_id=390 Accuracy=79.36: 100%|██████████| 391/391 [00:31<00:00, 12.53it/s]
Test set: Average loss: 0.6268, Accuracy: 7865/10000 (78.65%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 10
Train: Loss=0.5372 Batch_id=390 Accuracy=80.51: 100%|██████████| 391/391 [00:31<00:00, 12.44it/s]
Test set: Average loss: 0.6086, Accuracy: 8010/10000 (80.10%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 11
Train: Loss=0.5687 Batch_id=390 Accuracy=81.19: 100%|██████████| 391/391 [00:31<00:00, 12.51it/s]
Test set: Average loss: 0.6138, Accuracy: 7980/10000 (79.80%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 12
Train: Loss=0.4351 Batch_id=390 Accuracy=81.85: 100%|██████████| 391/391 [00:30<00:00, 12.84it/s]
Test set: Average loss: 0.5513, Accuracy: 8148/10000 (81.48%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 13
Train: Loss=0.3758 Batch_id=390 Accuracy=82.47: 100%|██████████| 391/391 [00:31<00:00, 12.56it/s]
Test set: Average loss: 0.5573, Accuracy: 8218/10000 (82.18%)

Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 14
Train: Loss=0.5081 Batch_id=390 Accuracy=82.88: 100%|██████████| 391/391 [00:31<00:00, 12.57it/s]
Test set: Average loss: 0.5824, Accuracy: 8106/10000 (81.06%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 15
Train: Loss=0.4065 Batch_id=390 Accuracy=85.81: 100%|██████████| 391/391 [00:31<00:00, 12.60it/s]
Test set: Average loss: 0.4541, Accuracy: 8474/10000 (84.74%)

Epoch 16
Train: Loss=0.2946 Batch_id=390 Accuracy=87.09: 100%|██████████| 391/391 [00:30<00:00, 12.61it/s]
Test set: Average loss: 0.4503, Accuracy: 8518/10000 (85.18%)

Epoch 17
Train: Loss=0.3998 Batch_id=390 Accuracy=87.31: 100%|██████████| 391/391 [00:32<00:00, 12.01it/s]
Test set: Average loss: 0.4438, Accuracy: 8566/10000 (85.66%)

Epoch 18
Train: Loss=0.2772 Batch_id=390 Accuracy=87.56: 100%|██████████| 391/391 [00:31<00:00, 12.48it/s]
Test set: Average loss: 0.4406, Accuracy: 8556/10000 (85.56%)

Epoch 19
Train: Loss=0.3993 Batch_id=390 Accuracy=87.60: 100%|██████████| 391/391 [00:32<00:00, 12.16it/s]
Test set: Average loss: 0.4420, Accuracy: 8560/10000 (85.60%)

Epoch 20
Train: Loss=0.2813 Batch_id=390 Accuracy=87.90: 100%|██████████| 391/391 [00:31<00:00, 12.38it/s]
Test set: Average loss: 0.4453, Accuracy: 8551/10000 (85.51%)


```

##### <b>Train/Test Visualization</b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V3/main/Session8/assets/train_test_acc_loss_comp.png)

##### <b>10 Mis-classified Images </b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V3/main/Session8/assets/cifar10_s8_final_misclassified_imgs.png)

