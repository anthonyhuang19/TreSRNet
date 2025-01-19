import torch
import torch.nn as nn
from torchvision import models


class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load VGG19 pretrained model and use only the first 23 layers
        self.vgg = models.vgg19(pretrained=True).features[:23].to(device).eval()
        # Freeze VGG parameters to avoid gradients during backpropagation
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Loss function (MSE loss)
        self.loss = nn.MSELoss()

        # Pre-trained VGG normalization parameters (mean and std values from ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, output, target):
        # Normalize the images to match VGG's training data (ImageNet)
        output = (output - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Extract features from both the output and target images using VGG
        vgg_output_features = self.vgg(output)
        vgg_target_features = self.vgg(target)

        # Calculate MSE loss between the VGG features of output and target images
        loss = self.loss(vgg_output_features, vgg_target_features)
        return loss
