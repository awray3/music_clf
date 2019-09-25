"""
Model classes.

As a reminder, the inputs to this model are melspectrograms, which
are torch tensors of size (1, T, 128), where T is a constant that only 
depends on the default sample rate of 16kHz and the specified duration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN_Dwivedi(nn.Module):
    
    def __init__(self, num_conv_channels=56, kernel_size=5, momentum=0.9):
        super(CRNN_Dwivedi, self).__init__()
        
        self.conv1 = nn.Conv2d(1, num_conv_channels, kernel_size)
        self.conv2 = nn.Conv2d(1, )
        self.conv3 = nn.Conv2d()

