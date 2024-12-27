import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### Session 5

class Net(nn.Module):
    """This defines the structure of the NN.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self):
        """_summary_
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) 
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = F.relu(self.conv1(x))                # input_size = 28x28x1, output_size = 26x26x32, RF = 3x3
        x = F.relu(F.max_pool2d(self.conv2(x),2))  # input_size = 26x26x32, output_size = 12x12x64, RF = 6x6
        x = F.relu(self.conv3(x))                # input_size = 12x12x64, output_size = 10x10x128, RF = 10x10
        x = F.relu(F.max_pool2d(self.conv4(x),2))  # input_size = 10x10x128, output_size = 4x4x256, RF = 16x16
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

### Session 7

class Model_6(nn.Module):

    def __init__(self, dropout_value=0):
        super(Model_6, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # input_size = 28x28x1, output_size = 26x26x8, RF = 3x3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # input_size = 26x26x8, output_size = 24x24x12, RF = 5x5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 24x24x12, output_size = 24x24x10, RF = 5x5
        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 24x24x10, output_size = 12x12x10, RF = 6x6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) # input_size = 12x12x10, output_size = 10x10x14, RF = 10x10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size = 10x10x14, output_size = 8x8x16, RF = 14x14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        ) # input_size = 8x8x16, output_size = 6x6x20, RF = 18x18

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # input_size = 6x6x20, output_size = 1x1x20, RF = 28x28

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x20, output_size = 1x1x16, RF = 28x28

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x16, output_size = 1x1x10, RF = 28x28

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)

class Model_7(nn.Module):

    def __init__(self, dropout_value=0):
        super(Model_7, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # input_size = 28x28x1, output_size = 26x26x8, RF = 3x3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) # input_size = 26x26x8, output_size = 24x24x14, RF = 5x5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 24x24x14, output_size = 24x24x10, RF = 5x5
        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 24x24x10, output_size = 12x12x10, RF = 6x6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) # input_size = 12x12x10, output_size = 10x10x14, RF = 10x10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size = 10x10x14, output_size = 8x8x16, RF = 14x14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        ) # input_size = 8x8x16, output_size = 6x6x20, RF = 18x18

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # input_size = 6x6x20, output_size = 1x1x20, RF = 28x28

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x20, output_size = 1x1x16, RF = 28x28

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x16, output_size = 1x1x10, RF = 28x28

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)

### Session 8

#################### CIFAR 10 Classifiers ################################

##### Batch Norm

class CIFAR10_Classifier(nn.Module):

    def __init__(self, dropout_value=0):
        super(CIFAR10_Classifier, self).__init__()
        # Input Block C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size = 32x32x3, output_size = 32x32x16, RF = 3x3

        # CONVOLUTION BLOCK 1 C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut1 = nn.Sequential()

        # P1
        self.pool1 = nn.MaxPool2d(2, 2) 

        # CONVOLUTION BLOCK 2 C3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) 

        # C4
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=28, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(28),
            nn.Dropout(dropout_value)
        ) 

        # C5
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut2 = nn.Sequential()

        # P2
        self.pool2 = nn.MaxPool2d(2, 2) 

         # CONVOLUTION BLOCK 2 C7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        ) 

        # C8
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=26, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(26),
            nn.Dropout(dropout_value)
        ) 

        # C9
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=26, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        # OUTPUT BLOCK GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) 

        # C10
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.convblock1(x)
        y = self.convblock2(x)
        y = self.convblock3(y)
        y += self.shortcut1(x)
        y = F.relu(y)

        y = self.pool1(y)
        y1 = self.convblock4(y)
        y1 = self.convblock5(y1)
        y1 = self.convblock6(y1)
        y1 = self.convblock7(y1)
        y1 += self.shortcut2(y)
        y1 = F.relu(y1)

        y1 = self.pool2(y1)
        y1 = self.convblock8(y1)
        y1 = self.convblock9(y1)
        y1 = self.convblock10(y1)
        y1 = self.gap(y1)
        y1 = self.convblock11(y1)
        y1 = y1.view(-1, 10)

        return F.log_softmax(y1, dim=-1)

############### Group Normalization

# C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

class CIFAR10_Classifier_GN(nn.Module):

    def __init__(self, num_groups=4, dropout_value=0):
        super(CIFAR10_Classifier_GN, self).__init__()
        self.num_groups = num_groups
        # Input Block C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 16),
            nn.Dropout(dropout_value)
        ) # input_size = 32x32x3, output_size = 32x32x16, RF = 3x3

        # CONVOLUTION BLOCK 1 C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 32),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut1 = nn.Sequential()

        # P1
        self.pool1 = nn.MaxPool2d(2, 2) 

        # CONVOLUTION BLOCK 2 C3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 24),
            nn.Dropout(dropout_value)
        ) 

        # C4
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=28, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 28),
            nn.Dropout(dropout_value)
        ) 

        # C5
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 32),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut2 = nn.Sequential()

        # P2
        self.pool2 = nn.MaxPool2d(2, 2) 

         # CONVOLUTION BLOCK 2 C7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 20),
            nn.Dropout(dropout_value)
        ) 

        # C8
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=28, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 28),
            nn.Dropout(dropout_value)
        )

        # C9
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(self.num_groups, 32),
            nn.Dropout(dropout_value)
        ) 

        # OUTPUT BLOCK GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) 

        # C10
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.convblock1(x)
        y = self.convblock2(x)
        y = self.convblock3(y)
        y += self.shortcut1(x)
        y = F.relu(y)

        y = self.pool1(y)
        y1 = self.convblock4(y)
        y1 = self.convblock5(y1)
        y1 = self.convblock6(y1)
        y1 = self.convblock7(y1)
        y1 += self.shortcut2(y)
        y1 = F.relu(y1)

        y1 = self.pool2(y1)
        y1 = self.convblock8(y1)
        y1 = self.convblock9(y1)
        y1 = self.convblock10(y1)
        y1 = self.gap(y1)
        y1 = self.convblock11(y1)
        y1 = y1.view(-1, 10)

        return F.log_softmax(y1, dim=-1)

###### Layer Normalization 

# C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

class CIFAR10_Classifier_LN(nn.Module):

    def __init__(self, dropout_value=0):
        super(CIFAR10_Classifier_LN, self).__init__()
        # Input Block C1
        self.convblock1 = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False, groups=3),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([8,32,32]),
            nn.Dropout(dropout_value)
        ) # input_size = 32x32x3, output_size = 32x32x8, RF = 3x3

        # CONVOLUTION BLOCK 1 C2
        self.convblock2 = nn.Sequential(
            # nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False, groups=8),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([10,32,32]),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 30x30x128, output_size = 30x30x128, RF = 5x5
        self.shortcut1 = nn.Sequential()

        # P1
        self.pool1 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 2 C3
        self.convblock4 = nn.Sequential(
            # nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False, groups=8),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([10,16,16]),
            nn.Dropout(dropout_value)
        ) 

        # C4
        self.convblock5 = nn.Sequential(
            # nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False, groups=10),
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([12,16,16]),
            nn.Dropout(dropout_value)
        )

        # C5
        self.convblock6 = nn.Sequential(
            # nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False, groups=12),
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([14,16,16]),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut2 = nn.Sequential()

        # P2
        self.pool2 = nn.MaxPool2d(2, 2) 

         # CONVOLUTION BLOCK 2 C7
        self.convblock8 = nn.Sequential(
            # nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False, groups=8),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([10,8,8]),
            nn.Dropout(dropout_value)
        )

        # C8
        self.convblock9 = nn.Sequential(
            # nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False, groups=10),
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([12,8,8]),
            nn.Dropout(dropout_value)
        )

        # C9
        self.convblock10 = nn.Sequential(
            # nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False, groups=12),
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([14,8,8]),
            nn.Dropout(dropout_value)
        ) 

        # OUTPUT BLOCK GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        # C10
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.convblock1(x)
        y = self.convblock2(x)
        y = self.convblock3(y)
        y += self.shortcut1(x)
        y = F.relu(y)

        y = self.pool1(y)
        y1 = self.convblock4(y)
        y1 = self.convblock5(y1)
        y1 = self.convblock6(y1)
        y1 = self.convblock7(y1)
        y1 += self.shortcut2(y)
        y1 = F.relu(y1)

        y1 = self.pool2(y1)
        y1 = self.convblock8(y1)
        y1 = self.convblock9(y1)
        y1 = self.convblock10(y1)
        y1 = self.gap(y1)
        y1 = self.convblock11(y1)
        y1 = y1.view(-1, 10)

        return F.log_softmax(y1, dim=-1)
    
class CIFAR10_Classifier_LN_Modified(nn.Module):

    def __init__(self, dropout_value=0):
        super(CIFAR10_Classifier_LN, self).__init__()
        # Input Block C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([8,30,30]),
            nn.Dropout(dropout_value)
        ) # input_size = 32x32x3, output_size = 32x32x8, RF = 3x3

        # CONVOLUTION BLOCK 1 C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([10,28,28]),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.shortcut1 = nn.Sequential()

        # P1
        self.pool1 = nn.MaxPool2d(2, 2) 

        # CONVOLUTION BLOCK 2 C3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([10,14,14]),
            nn.Dropout(dropout_value)
        ) 

        # C4
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([12,14,14]),
            nn.Dropout(dropout_value)
        )

        # C5
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([14,14,14]),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1 c6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut2 = nn.Sequential()

        # P2
        self.pool2 = nn.MaxPool2d(2, 2) 

         # CONVOLUTION BLOCK 2 C7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([10,7,7]),
            nn.Dropout(dropout_value)
        ) 

        # C8
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([12,7,7]),
            nn.Dropout(dropout_value)
        ) 

        # C9
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([14,7,7]),
            nn.Dropout(dropout_value)
        ) 

        # OUTPUT BLOCK GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        ) 

        # C10
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.convblock1(x)
        y = self.convblock2(x)
        y = self.convblock3(y)
        # y += self.shortcut1(x)
        # y = F.relu(y)

        y = self.pool1(y)
        y1 = self.convblock4(y)
        y1 = self.convblock5(y1)
        y1 = self.convblock6(y1)
        y1 = self.convblock7(y1)
        y1 += self.shortcut2(y)
        y1 = F.relu(y1)

        y1 = self.pool2(y1)
        y1 = self.convblock8(y1)
        y1 = self.convblock9(y1)
        y1 = self.convblock10(y1)
        y1 = self.gap(y1)
        y1 = self.convblock11(y1)
        y1 = y1.view(-1, 10)

        return F.log_softmax(y1, dim=-1)
    
################################################################################

### Session 9 Assignment

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class CIFAR10_NoMPNetwork(nn.Module):

    def __init__(self, dropout_value=0):
        super(CIFAR10_NoMPNetwork, self).__init__()
        # Input Block 
        self.inpblock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        # C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )

        # TRANSITION BLOCK 1 T1
        self.transitionblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut1 = nn.Sequential()

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        ) # input_size = 32x32x3, output_size = 32x32x16, RF = 3x3

        # CONVOLUTION BLOCK 2 C2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=62, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(62),
            nn.Dropout(dropout_value),
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=62, out_channels=72, kernel_size=(3, 3), padding=1, bias=False),
            # SeparableConv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.Dropout(dropout_value),
        )

        # TRANSITION BLOCK 2 T2
        self.transitionblock2 = nn.Sequential(
            nn.Conv2d(in_channels=72, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.shortcut2 = nn.Sequential()

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        # CONVOLUTION BLOCK 3 C3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            # SeparableConv2d(in_channels=32, out_channels=42, kernel_size=1, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            # nn.Conv2d(in_channels=86, out_channels=100, kernel_size=(3, 3), padding=1, bias=False),
            SeparableConv2d(in_channels=64, out_channels=84, kernel_size=1, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(84),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(84),
            nn.Dropout(dropout_value)
        ) 

        # OUTPUT BLOCK 
        self.output = nn.Sequential(
            nn.AvgPool2d(kernel_size=20),
            nn.Conv2d(in_channels=84, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.inpblock(x)
        z = self.convblock1(x)
        z = self.convblock2(z)
        z = self.transitionblock1(z)
        z += self.shortcut1(x)
        z = F.relu(z)
        z = self.convblock3(z)

        y = self.convblock4(z)
        y = self.convblock5(y)
        y = self.transitionblock2(y)
        y += self.shortcut2(z)
        y = F.relu(y)
        y = self.convblock6(y)

        y = self.convblock7(y)
        y = self.output(y)
        y = y.view(-1, 10)

        return F.log_softmax(y, dim=-1)

class CIFAR10_NoMPNetwork_Modified(nn.Module):

    def __init__(self, dropout_value=0):
        super(CIFAR10_NoMPNetwork_Modified, self).__init__()
        # Input Block
        self.inpblock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        # C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )

        # TRANSITION BLOCK 1 T1
        self.transitionblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.shortcut1 = nn.Sequential()

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        ) # input_size = 32x32x3, output_size = 32x32x16, RF = 3x3

        # CONVOLUTION BLOCK 2 C2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=62, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(62),
            nn.Dropout(dropout_value),
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=62, out_channels=72, kernel_size=(3, 3), padding=1, bias=False),
            # SeparableConv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.Dropout(dropout_value),
        )

        # TRANSITION BLOCK 2 T2
        self.transitionblock2 = nn.Sequential(
            nn.Conv2d(in_channels=72, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.shortcut2 = nn.Sequential()

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 3 C3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            # SeparableConv2d(in_channels=32, out_channels=42, kernel_size=1, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            # nn.Conv2d(in_channels=86, out_channels=100, kernel_size=(3, 3), padding=1, bias=False),
            SeparableConv2d(in_channels=64, out_channels=84, kernel_size=1, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(84),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(3, 3), padding=0, bias=False, dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(84),
            nn.Dropout(dropout_value)
        )

        # OUTPUT BLOCK
        self.output = nn.Sequential(
            nn.AvgPool2d(kernel_size=14),
            nn.Conv2d(in_channels=84, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.inpblock(x)
        z = self.convblock1(x)
        z = self.convblock2(z)
        z = self.transitionblock1(z)
        z += self.shortcut1(x)
        z = F.relu(z)
        z = self.convblock3(z)

        y = self.convblock4(z)
        y = self.convblock5(y)
        y = self.transitionblock2(y)
        y += self.shortcut2(z)
        y = F.relu(y)
        y = self.convblock6(y)

        y = self.convblock7(y)
        y = self.output(y)
        y = y.view(-1, 10)

        return F.log_softmax(y, dim=-1)