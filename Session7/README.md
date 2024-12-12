# MNIST-trained Model with 99.4% test accuracy in < 8K Parameters and <15 epochs

##### Trial Summary

| S.No. | File Name | Highlight |Targets | Results | Analysis | File Link |
|---|---|---|---|---|---|---|
|1|S7_File1 | Basic Skeleton Creation|Create a basic skeleton with less than 8K parameters which is able to reach 99% in less than 15 epochs. The basic skeleton was created based on the expand and squeeze architecture.|<ul><li>Best Train Accuracy - 99.59%</li><li> Best Test Accuracy - 99.27%</li><li>Total Parameters - 8000</li></ul>|Good starting model but high overfitting|[Open](https://github.com/garima-mahato/ERA_V3/blob/main/Session7/S7_File1.ipynb)|
|2|S7_File2 | Improving the Basic Model (Reducing Overfitting)| Improve the basic model by reducing overfitting. Added dropout of 0.05 to reduce overfitting. With the basic model, I was able to achieve 99.4% accuracy when trained for around 40 epochs. This means that the model has the capacity to reach 99.4%. So, after adding dropout, I added step LR starting at 0.1 and reducing by 0.1 at every 4 epochs. These 2 numbers were found after experimenting.|<ul><li>Best Train Accuracy - 98.54%</li><li> Best Test Accuracy - 99.38%</li><li>Total Parameters - 8000</li></ul>|Overfitting reduced and was consistently able to maintain 99.3% test accuracy. Increasing the learning rate to 0.1 helped to reach higher accuracy sooner and gradually decreasing the learning rate by 0.1 helped in achieving stable results. Giving more training sample can improve the learning of the model.|[Open](https://github.com/garima-mahato/ERA_V3/blob/main/Session7/S7_File2.ipynb)|
|3|S7_File3 | Improving the Model (Image Augmentation, Batch Size(Sweet Spot), Regularization)|Improve the model learning by:<ol><li>Adding image augmentation</li><li>Reducing batch size</li><li>Adding regularization at correct position with reduced batch size</li><ol>|<ul><li>Best Train Accuracy - 98.55%</li><li> Best Test Accuracy - 99.42%</li><li>Total Parameters - 8000</li></ul>|<ol><li>Adding image augmentation of scaling, translation and rotation increased the difficulty of model's training so we see an improvement in the test accuracy</li><li>Reducing batch size from 512 to 128 improved the generalization capability of the model on the test dataset and brought the test accuracy in the 99.4% threshold. 128 batch size is the sweet spot for this model, below which the test accuracy degrades. This is due to the existence to “noise” in small batch size training. Because neural network systems are extremely prone to overfitting, upon seeing many small batch size, each batch being a “noisy” representation of the entire dataset, will cause a sort of “tug-and-pull” dynamic. This “tug-and-pull” dynamic prevents the neural network from overfitting on the training set and hence performing badly on the test set.</li><li>Adding STEP LR at correct position of 8 epochs instead of 4. This helped in reducing the epochs for achieving 99.4% test accuracy consistently.</li><ol>With the above experimentation, I was able to achieve 99.4% test accuracy consistently.|[Open](https://github.com/garima-mahato/ERA_V3/blob/main/Session7/S7_File3.ipynb)|

---

### Final Model Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 14, 24, 24]           1,008
              ReLU-6           [-1, 14, 24, 24]               0
       BatchNorm2d-7           [-1, 14, 24, 24]              28
           Dropout-8           [-1, 14, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             140
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 14, 10, 10]           1,260
             ReLU-12           [-1, 14, 10, 10]               0
      BatchNorm2d-13           [-1, 14, 10, 10]              28
          Dropout-14           [-1, 14, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,016
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 20, 6, 6]           2,880
             ReLU-20             [-1, 20, 6, 6]               0
      BatchNorm2d-21             [-1, 20, 6, 6]              40
          Dropout-22             [-1, 20, 6, 6]               0
        AvgPool2d-23             [-1, 20, 1, 1]               0
           Conv2d-24             [-1, 16, 1, 1]             320
           Conv2d-25             [-1, 10, 1, 1]             160
================================================================
Total params: 8,000
Trainable params: 8,000
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.56
Params size (MB): 0.03
Estimated Total Size (MB): 0.60
----------------------------------------------------------------
```

| Block | Layer | Input Size | Output Size | Receptive Field |
|---|---|---|---|---|
| Input Block | Conv2D(3x3) | 28x28x1 | 26x26x8 | 3x3 |
| Convolution Block 1 | Conv2D(3x3) | 26x26x8 | 24x24x14 | 5x5 |
| Transition Block 1 | Conv2D(1x1) | 24x24x14 | 24x24x10 | 5x5 |
| Convolution Block 1 | Max Pool(2x2) | 24x24x10 | 12x12x10 | 6x6 |
| Convolution Block 1 | Conv2D(3x3) | 12x12x10 | 10x10x14 | 10x10 |
| Convolution Block 1 | Conv2D(3x3) | 10x10x14 | 8x8x16 | 14x14 |
| Convolution Block 1 | Conv2D(3x3) | 8x8x16 | 6x6x20 | 18x18 |
| Output Block | GAP | 6x6x20 | 1x1x20 | 28x28 |
| Output Block | FC | 1x1x20 | 1x1x16 | 28x28 |
| Output Block | FC | 1x1x16 | 1x1x10 | 28x28 | 

##### Final Results: 
  - Best Train Accuracy - 98.55%
  - Best Test Accuracy - 99.42%
  - Total Parameters - 8000

---

# Trials Details

## S7_File1: Basic Skeleton Creation

##### Targets: 
Create a basic skeleton with less than 8K parameters which is able to reach 99% in less than 15 epochs. The basic skeleton was created based on the expand and squeeze architecture.

##### Results: 
  - Best Train Accuracy - 99.59%
  - Best Test Accuracy - 99.27%
  - Total Parameters - 8000

##### Analysis: 
Good starting model but high overfitting

<b>Train/Test Logs</b>

```
Epoch 1
Train: Loss=0.0996 Batch_id=117 Accuracy=85.17: 100%|██████████| 118/118 [00:21<00:00,  5.46it/s]
Test set: Average loss: 0.1631, Accuracy: 9509/10000 (95.09%)

Epoch 2
Train: Loss=0.0364 Batch_id=117 Accuracy=98.00: 100%|██████████| 118/118 [00:21<00:00,  5.46it/s]
Test set: Average loss: 0.0541, Accuracy: 9833/10000 (98.33%)

Epoch 3
Train: Loss=0.1695 Batch_id=117 Accuracy=98.56: 100%|██████████| 118/118 [00:21<00:00,  5.38it/s]
Test set: Average loss: 0.0441, Accuracy: 9854/10000 (98.54%)

Epoch 4
Train: Loss=0.0391 Batch_id=117 Accuracy=98.78: 100%|██████████| 118/118 [00:21<00:00,  5.48it/s]
Test set: Average loss: 0.0378, Accuracy: 9893/10000 (98.93%)

Epoch 5
Train: Loss=0.0258 Batch_id=117 Accuracy=99.02: 100%|██████████| 118/118 [00:22<00:00,  5.31it/s]
Test set: Average loss: 0.0381, Accuracy: 9889/10000 (98.89%)

Epoch 6
Train: Loss=0.0136 Batch_id=117 Accuracy=99.01: 100%|██████████| 118/118 [00:21<00:00,  5.60it/s]
Test set: Average loss: 0.0357, Accuracy: 9879/10000 (98.79%)

Epoch 7
Train: Loss=0.0276 Batch_id=117 Accuracy=99.21: 100%|██████████| 118/118 [00:20<00:00,  5.68it/s]
Test set: Average loss: 0.0299, Accuracy: 9908/10000 (99.08%)

Epoch 8
Train: Loss=0.0548 Batch_id=117 Accuracy=99.24: 100%|██████████| 118/118 [00:20<00:00,  5.70it/s]
Test set: Average loss: 0.0287, Accuracy: 9908/10000 (99.08%)

Epoch 9
Train: Loss=0.0077 Batch_id=117 Accuracy=99.28: 100%|██████████| 118/118 [00:20<00:00,  5.70it/s]
Test set: Average loss: 0.0318, Accuracy: 9897/10000 (98.97%)

Epoch 10
Train: Loss=0.0552 Batch_id=117 Accuracy=99.36: 100%|██████████| 118/118 [00:21<00:00,  5.49it/s]
Test set: Average loss: 0.0251, Accuracy: 9916/10000 (99.16%)

Epoch 11
Train: Loss=0.0032 Batch_id=117 Accuracy=99.43: 100%|██████████| 118/118 [00:22<00:00,  5.24it/s]
Test set: Average loss: 0.0248, Accuracy: 9920/10000 (99.20%)

Epoch 12
Train: Loss=0.0199 Batch_id=117 Accuracy=99.53: 100%|██████████| 118/118 [00:21<00:00,  5.47it/s]
Test set: Average loss: 0.0271, Accuracy: 9926/10000 (99.26%)

Epoch 13
Train: Loss=0.0038 Batch_id=117 Accuracy=99.55: 100%|██████████| 118/118 [00:21<00:00,  5.45it/s]
Test set: Average loss: 0.0308, Accuracy: 9912/10000 (99.12%)

Epoch 14
Train: Loss=0.0014 Batch_id=117 Accuracy=99.59: 100%|██████████| 118/118 [00:21<00:00,  5.48it/s]
Test set: Average loss: 0.0259, Accuracy: 9917/10000 (99.17%)

Epoch 15
Train: Loss=0.0230 Batch_id=117 Accuracy=99.58: 100%|██████████| 118/118 [00:21<00:00,  5.39it/s]
Test set: Average loss: 0.0247, Accuracy: 9927/10000 (99.27%)

```

<b>Train/Test Visualization</b>

![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session7_InDepthCodingPractice/assets/s7_file1_train_test_acc_loss.png)

---

## S7_File2: Improving the Basic Model (Reducing Overfitting)

##### Targets: 
  Improve the basic model by reducing overfitting. Added dropout of 0.05 to reduce overfitting. 
  With the basic model, I was able to achieve 99.4% accuracy when trained for around 40 epochs. This means that the model has the capacity to reach 99.4%. So, after adding dropout, I added step LR starting at 0.1 and reducing by 0.1 at every 4 epochs. These 2 numbers were found after experimenting.

##### Results: 
  - Best Train Accuracy - 98.54%
  - Best Test Accuracy - 99.38%
  - Total Parameters - 8000

##### Analysis: 
Overfitting reduced and was consistently able to maintain 99.3% test accuracy. Increasing the learning rate to 0.1 helped to reach higher accuracy sooner and gradually decreasing the learning rate by 0.1 helped in achieving stable results. Giving more training sample can improve the learning of the model.

<b>Train/Test Logs</b>

```
Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 1
Train: Loss=0.1322 Batch_id=468 Accuracy=89.18: 100%|██████████| 469/469 [00:32<00:00, 14.39it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0502, Accuracy: 9838/10000 (98.38%)

Epoch 2
Train: Loss=0.1159 Batch_id=468 Accuracy=96.72: 100%|██████████| 469/469 [00:31<00:00, 14.85it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0327, Accuracy: 9890/10000 (98.90%)

Epoch 3
Train: Loss=0.0532 Batch_id=468 Accuracy=97.26: 100%|██████████| 469/469 [00:33<00:00, 14.17it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0286, Accuracy: 9917/10000 (99.17%)

Epoch 4
Train: Loss=0.0535 Batch_id=468 Accuracy=97.46: 100%|██████████| 469/469 [00:33<00:00, 13.80it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0318, Accuracy: 9899/10000 (98.99%)

Epoch 5
Train: Loss=0.1066 Batch_id=468 Accuracy=98.14: 100%|██████████| 469/469 [00:32<00:00, 14.22it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0202, Accuracy: 9933/10000 (99.33%)

Epoch 6
Train: Loss=0.0813 Batch_id=468 Accuracy=98.30: 100%|██████████| 469/469 [00:32<00:00, 14.54it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0199, Accuracy: 9935/10000 (99.35%)

Epoch 7
Train: Loss=0.0108 Batch_id=468 Accuracy=98.33: 100%|██████████| 469/469 [00:32<00:00, 14.34it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0200, Accuracy: 9934/10000 (99.34%)

Epoch 8
Train: Loss=0.0686 Batch_id=468 Accuracy=98.32: 100%|██████████| 469/469 [00:34<00:00, 13.68it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0189, Accuracy: 9937/10000 (99.37%)

Epoch 9
Train: Loss=0.0230 Batch_id=468 Accuracy=98.50: 100%|██████████| 469/469 [00:32<00:00, 14.25it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0193, Accuracy: 9936/10000 (99.36%)

Epoch 10
Train: Loss=0.0431 Batch_id=468 Accuracy=98.40: 100%|██████████| 469/469 [00:31<00:00, 14.76it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0197, Accuracy: 9932/10000 (99.32%)

Epoch 11
Train: Loss=0.0999 Batch_id=468 Accuracy=98.38: 100%|██████████| 469/469 [00:31<00:00, 14.85it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0190, Accuracy: 9936/10000 (99.36%)

Epoch 12
Train: Loss=0.1079 Batch_id=468 Accuracy=98.53: 100%|██████████| 469/469 [00:33<00:00, 14.07it/s]Adjusting learning rate of group 0 to 1.0000e-04.

Test set: Average loss: 0.0188, Accuracy: 9938/10000 (99.38%)

Epoch 13
Train: Loss=0.0809 Batch_id=468 Accuracy=98.54: 100%|██████████| 469/469 [00:32<00:00, 14.48it/s]Adjusting learning rate of group 0 to 1.0000e-04.

Test set: Average loss: 0.0196, Accuracy: 9935/10000 (99.35%)

Epoch 14
Train: Loss=0.0260 Batch_id=468 Accuracy=98.53: 100%|██████████| 469/469 [00:31<00:00, 14.93it/s]Adjusting learning rate of group 0 to 1.0000e-04.

Test set: Average loss: 0.0189, Accuracy: 9937/10000 (99.37%)

Epoch 15
Train: Loss=0.0853 Batch_id=468 Accuracy=98.45: 100%|██████████| 469/469 [00:31<00:00, 15.00it/s]Adjusting learning rate of group 0 to 1.0000e-04.

Test set: Average loss: 0.0198, Accuracy: 9933/10000 (99.33%)

```


---

## S7_File3: Improving the Model (Image Augmentation, Batch Size(Sweet Spot), Regularization)


##### Targets: 
Improve the model learning by:

  i) Adding image augmentation

  ii) Reducing batch size

  iii) Adding regularization at correct position with reduced batch size
  
##### Results: 
  - Best Train Accuracy - 98.55%
  - Best Test Accuracy - 99.42%
  - Total Parameters - 8000

##### Analysis: 
  i) Adding image augmentation of scaling, translation and rotation increased the difficulty of model's training so we see an improvement in the test accuracy

  ii) Reducing batch size from 512 to 128 improved the generalization capability of the model on the test dataset and brought the test accuracy in the 99.4% threshold. 128 batch size is the sweet spot for this model, below which the test accuracy degrades. This is due to the existence to “noise” in small batch size training. Because neural network systems are extremely prone to overfitting, upon seeing many small batch size, each batch being a “noisy” representation of the entire dataset, will cause a sort of “tug-and-pull” dynamic. This “tug-and-pull” dynamic prevents the neural network from overfitting on the training set and hence performing badly on the test set.

  iii) Adding STEP LR at correct position of 8 epochs instead of 4. This helped in reducing the epochs for achieving 99.4% test accuracy consistently.

  With the above experimentation, I was able to achieve 99.4% test accuracy consistently.

  <b>Train/Test Logs</b>

  ```
  Adjusting learning rate of group 0 to 1.0000e-01.
Epoch 1
Train: Loss=0.0786 Batch_id=468 Accuracy=87.56: 100%|██████████| 469/469 [00:33<00:00, 14.01it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0528, Accuracy: 9831/10000 (98.31%)

Epoch 2
Train: Loss=0.1295 Batch_id=468 Accuracy=96.28: 100%|██████████| 469/469 [00:25<00:00, 18.15it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0507, Accuracy: 9841/10000 (98.41%)

Epoch 3
Train: Loss=0.1093 Batch_id=468 Accuracy=97.08: 100%|██████████| 469/469 [00:26<00:00, 18.00it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0531, Accuracy: 9841/10000 (98.41%)

Epoch 4
Train: Loss=0.1115 Batch_id=468 Accuracy=97.30: 100%|██████████| 469/469 [00:27<00:00, 17.20it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0295, Accuracy: 9905/10000 (99.05%)

Epoch 5
Train: Loss=0.0523 Batch_id=468 Accuracy=97.48: 100%|██████████| 469/469 [00:26<00:00, 17.98it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0334, Accuracy: 9895/10000 (98.95%)

Epoch 6
Train: Loss=0.0316 Batch_id=468 Accuracy=97.72: 100%|██████████| 469/469 [00:25<00:00, 18.20it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0367, Accuracy: 9884/10000 (98.84%)

Epoch 7
Train: Loss=0.0307 Batch_id=468 Accuracy=97.87: 100%|██████████| 469/469 [00:26<00:00, 17.84it/s]Adjusting learning rate of group 0 to 1.0000e-01.

Test set: Average loss: 0.0242, Accuracy: 9919/10000 (99.19%)

Epoch 8
Train: Loss=0.0396 Batch_id=468 Accuracy=97.92: 100%|██████████| 469/469 [00:25<00:00, 18.06it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0372, Accuracy: 9878/10000 (98.78%)

Epoch 9
Train: Loss=0.0392 Batch_id=468 Accuracy=98.36: 100%|██████████| 469/469 [00:26<00:00, 17.79it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0190, Accuracy: 9936/10000 (99.36%)

Epoch 10
Train: Loss=0.0800 Batch_id=468 Accuracy=98.34: 100%|██████████| 469/469 [00:26<00:00, 17.44it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0186, Accuracy: 9942/10000 (99.42%)

Epoch 11
Train: Loss=0.0219 Batch_id=468 Accuracy=98.45: 100%|██████████| 469/469 [00:26<00:00, 17.87it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0189, Accuracy: 9942/10000 (99.42%)

Epoch 12
Train: Loss=0.0387 Batch_id=468 Accuracy=98.50: 100%|██████████| 469/469 [00:25<00:00, 18.15it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0182, Accuracy: 9940/10000 (99.40%)

Epoch 13
Train: Loss=0.0534 Batch_id=468 Accuracy=98.46: 100%|██████████| 469/469 [00:26<00:00, 18.03it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0186, Accuracy: 9940/10000 (99.40%)

Epoch 14
Train: Loss=0.0075 Batch_id=468 Accuracy=98.55: 100%|██████████| 469/469 [00:26<00:00, 17.68it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0184, Accuracy: 9941/10000 (99.41%)

Epoch 15
Train: Loss=0.0594 Batch_id=468 Accuracy=98.48: 100%|██████████| 469/469 [00:26<00:00, 17.49it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0183, Accuracy: 9942/10000 (99.42%)
  ```

  <b>Train/Test Visualization</b>

  ![](https://raw.githubusercontent.com/garima-mahato/ERA_V1/main/Session7_InDepthCodingPractice/assets/s7_file3_train_test_acc_loss.png)
