import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        if len(hidden_features) == 0:
            blocks.append(Linear(in_features, num_classes))
           
        else:
            blocks.append(Linear(in_features, hidden_features[0]))
            
            for i in range(len(hidden_features) - 1):
                if activation is 'relu':
                    blocks.append(ReLU())
                else:
                    blocks.append(Sigmoid())
                if(dropout> 0):
                    blocks.append(Dropout(dropout))
                blocks.append(Linear(hidden_features[i], hidden_features[i+1]))
            
            if activation is 'relu':
                blocks.append(ReLU())
            else:
                blocks.append(Sigmoid())
            if(dropout> 0):
                blocks.append(Dropout(dropout))
            
            blocks.append(Linear(hidden_features[-1], num_classes))
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        self.h = in_h
        self.w = in_w

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        self.filters.insert(0, in_channels)  # at the beginning, apply filter of each channel

        # 3x3 convolutions settings
        kernel_size_conv = (3, 3)
        stride_conv = (1, 1)
        padding_conv = (1, 1)

        # 2x2 max pooling settings
        kernel_size_pool = (2, 2)
        stride_pool = (2, 2)
        padding_pool = (0, 0)

        #  [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
        for i in range(N // P):
            for j in range(P):
                layers.append(nn.Conv2d(self.filters[i * P + j], self.filters[i * P + j + 1], kernel_size=kernel_size_conv,stride=stride_conv, padding=padding_conv))
                layers.append(nn.ReLU(inplace=True))
                self.h = (self.h - kernel_size_conv[0] + 2 * padding_conv[0]) // stride_conv[0] + 1  # W' = (W-F+2P)/S+1
                self.w = (self.w - kernel_size_conv[1] + 2 * padding_conv[1]) // stride_conv[1] + 1  # H' = (H-F+2P)/S+1

            layers.append(nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool, padding=padding_pool))
            self.h = (self.h - kernel_size_pool[0] + 2 * padding_pool[0]) // stride_pool[0] + 1
            self.w = (self.w - kernel_size_pool[1] + 2 * padding_pool[1]) // stride_pool[1] + 1

        self.filters.pop(0)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        layers.append(nn.Flatten())
        M = len(self.hidden_dims)

        self.hidden_dims.insert(0, int(self.filters[-1] * self.h * self.w))

        for i in range(M):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(self.hidden_dims[M], self.out_classes))
        self.hidden_dims.pop(0)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        out = self.classifier(features)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    class ResidualBlock(nn.Module):
        def __init__(self, filters, *args, **kwargs):
            super().__init__()
            self.filters = filters
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters[0], filters[-1], kernel_size=1, stride=(1, 1)),
                nn.BatchNorm2d(filters[-1], track_running_stats=False))

            stride_conv = (1, 1)
            kernel_size_conv = (3, 3)
            padding_conv = (1, 1)

            layers = []

            for i in range(len(filters) - 1):
                if i != 0:
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(p=0.5))

                layers.append(torch.nn.Conv2d(filters[i], filters[i + 1], stride=stride_conv, padding=padding_conv, kernel_size=kernel_size_conv))
                layers.append(nn.BatchNorm2d(filters[i + 1]))

            self.block_seq = nn.Sequential(*layers)

        def forward(self, x):
            if self.filters[0] != self.filters[-1]:
                residual = self.shortcut(x)
            else:
                residual = x
            out = self.block_seq(x)
            out += residual
            out = F.relu(out)
            return out

    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        self.h = in_h
        self.w = in_w

        layers = []

        N = len(self.filters)
        P = self.pool_every
        C = self.in_size[0]
        filters = self.filters
        filters.insert(0, C)

        padding = (0, 0)
        kernel_size = (2, 2)
        stride = (2, 2)
        k = 0

        stride_conv = (1, 1)
        kernel_size_conv = (3, 3)
        padding_conv = (1, 1)

        for i in range(N // P):
            for j in range(0, P - 1, 2):
                layers.append(YourCodeNet.ResidualBlock(filters[k:k + 3]))
                self.h = (self.h - kernel_size_conv[0] + 2 * padding_conv[0]) // stride_conv[0] + 1  # W' = (W-F+2P)/S+1
                self.w = (self.w - kernel_size_conv[1] + 2 * padding_conv[1]) // stride_conv[1] + 1  # H' = (H-F+2P)/S+1
                self.h = (self.h - kernel_size_conv[0] + 2 * padding_conv[0]) // stride_conv[0] + 1  # W' = (W-F+2P)/S+1
                self.w = (self.w - kernel_size_conv[1] + 2 * padding_conv[1]) // stride_conv[1] + 1  # H' = (H-F+2P)/S+1
                k += 3

            layers.append(torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            self.h = (((self.h + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0]) + 1)
            self.w = (((self.w + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1]) + 1)

        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================
