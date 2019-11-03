"""
Model classes.

As a reminder, the inputs to this model are melspectrograms, which
are torch tensors of size (1, T, 128), where T is a constant that only 
depends on the default sample rate of 16kHz and the specified duration.
"""

import torch.nn as nn
import torch.nn.functional as F

class Baseline_cnn(nn.Module):

    """
    arxiv.org/1802.09697 for this model.
    """

    def __init__(self, momentum=0.9):
        super(Baseline_cnn, self).__init__()

        # input size: (1, 64, T)
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 5))
        self.dense1 = nn.Linear(64 * 14 * 4, 32)
        self.dense2 = nn.Linear(32, 8)

    def forward(self, x):
        # input size: (N, 1, 64, 87)
        # after conv1: (N, 64, 62, 85)
        x = F.max_pool2d((F.relu(self.conv1(x))), kernel_size=(2, 4))
        # size: (N, 64, 31, 21)
        x = F.relu(self.conv2(x))
        # size: (N, 64, 29, 17)
        x = F.max_pool2d(x, kernel_size=(2, 4))
        # size: (N, 64, 14, 4)
        x = F.dropout(x, p=0.2)

        # flatten the tensor to get ready for the upcoming linear layer
        x = x.view(-1, self.num_flat_features(x))
        # size: (2, 3584)
        x = F.dropout(F.relu(self.dense1(x)), p=0.2)
        # size: (2, 32)
        x = F.softmax(self.dense2(x), dim=1)
        # size: (2, 8)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features



if __name__=='__main__':

    import torch
    model = Baseline_cnn()
    x = torch.randn(1, 1, 64, 87)
    print(model(x))





# class CRNN_Dwivedi(nn.Module):
    # def __init__(self, num_conv_channels=56, kernel_size=5, momentum=0.9):
        # super(CRNN_Dwivedi, self).__init__()
        # self.conv1 = nn.Conv2d(1, num_conv_channels, kernel_size)
        # self.conv2 = nn.Conv2d(1, )
        # self.conv3 = nn.Conv2d()
