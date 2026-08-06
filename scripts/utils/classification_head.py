import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """Classification head"""
    def __init__(self,
                 num_classes,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 dropout: float = 0.3):
        super(ClassificationHead, self).__init__()
        
        self.out_channels = out_channels

        if self.out_channels is None:
            raise ValueError("out_channels cannot be None. Provide a list of output channels.")

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Output size: (B x C x 1 x 1 x 1)
        
        # Classification head
        layers = []
        
        # Create fully connected layers based on out_channels
        for i in range(len(self.out_channels) - 1):
            layers.append(nn.Linear(self.out_channels[i], self.out_channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None: layers.append(nn.Dropout(dropout))
        
        # Final layer for classification
        layers.append(nn.Linear(self.out_channels[-1], num_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # Global average pooling
        x = self.avg_pool(x)        # [B, C, 1, 1, 1]

        # Flatten the tensor to [batch_size, out_channels]
        x = x.view(x.size(0), -1)   # Flatten to [B, C]
        
        if x.shape[-1] != self.out_channels[0]:
          raise ValueError(f"ClassificationHead expected {x.shape[-1]} channels as input but received {self.out_channels[0]}. Check the out_channels parameter in the config file.")
        
        # Pass through fully connected layers
        x = self.fc(x)

        return x


#%% Example case
# model = ClassificationHead(num_classes=10, out_channels=[1024, 512, 256])
# input_tensor = torch.randn(1, 1024, 4, 7, 7)  # Example input
# output = model(input_tensor)

# print('Output shape:', output.shape)  # torch.Size([1, 10])

#%%
# =========================================================================================================
class ClassificationHeadWithoutFlatten(nn.Module):
    """Classification head without flattening. It assumes that the feature maps are already flatten.
       So, the tensor is a flattened tensor."""
    def __init__(self,
                 num_classes,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 dropout: float = 0.3):
        super(ClassificationHeadWithoutFlatten, self).__init__()
        
        self.out_channels = out_channels

        if self.out_channels is None:
            raise ValueError("out_channels cannot be None. Provide a list of output channels.")
        
        # Classification head
        layers = []
        
        # Create fully connected layers based on out_channels
        for i in range(len(self.out_channels) - 1):
            layers.append(nn.Linear(self.out_channels[i], self.out_channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None: layers.append(nn.Dropout(dropout))
        
        # Final layer for classification
        layers.append(nn.Linear(self.out_channels[-1], num_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        
        if x.shape[-1] != self.out_channels[0]:
          raise ValueError(f"ClassificationHead expected {x.shape[-1]} channels as input but received {self.out_channels[0]}. Check the out_channels parameter in the config file.")
        
        # Pass through fully connected layers
        x = self.fc(x)

        return x


# ======================================================================================
# Classification head for CTCLIP
class ClassificationHeadCTCLIP(nn.Module):
    """Classification head without flattening. It assumes that the feature maps are already flatten.
       So, the tensor is a flattened tensor."""
    def __init__(self,
                 num_classes,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 dropout: float = 0.3):
        super(ClassificationHeadCTCLIP, self).__init__()
        
        self.out_channels = out_channels

        if self.out_channels is None:
            raise ValueError("out_channels cannot be None. Provide a list of output channels.")
        
        # Classification head
        layers = []
        
        # Add num_classes to the out_channels
        self.out_channels = self.out_channels + [num_classes]

        # Create fully connected layers based on out_channels
        for i in range(len(self.out_channels) - 1):
            layers.append(nn.ReLU()) # removed inplace=True to make it consistant with CTCLIP
            if dropout is not None: layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(self.out_channels[i], self.out_channels[i + 1]))
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        
        if x.shape[-1] != self.out_channels[0]:
          raise ValueError(f"ClassificationHead expected {x.shape[-1]} channels as input but received {self.out_channels[0]}. Check the out_channels parameter in the config file.")
        
        # Pass through fully connected layers
        x = self.fc(x)

        return x