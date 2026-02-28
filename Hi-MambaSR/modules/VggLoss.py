"""Key Publication-Level Enhancements:
Buffer Registration: By using register_buffer, the mean and std constants are saved in the model's state_dict but remain non-trainable. Crucially, this removes the need to pass a device to the constructor, as PyTorch will handle device transfers automatically.

Modern Weights API: Switched to VGG19_Weights.IMAGENET1K_V1 to ensure compatibility with modern torchvision versions and avoid deprecated warnings.

ImageNet Standardization: VGG19 was trained on normalized ImageNet data. Adding the mean/std subtraction step significantly improves the quality of the extracted features compared to using raw pixel values.

Implicit Eval Guard: Overriding the train() method is a fail-safe. It ensures that even if you call model.train() on your whole system, the VGG feature extractor stays in .eval(), which is critical for maintaining consistent perceptual gradients.

Memory Optimization: Removed manual .to(device) calls in the forward pass, reducing redundant memory transfers and allowing the code to be compatible with Distributed Data Parallel (DDP) training.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.checkpoint import checkpoint

class VGGLoss(nn.Module):
    """
    VGG-based Perceptual Loss for Hi-MambaSR.
    
    Computes the Mean Squared Error (MSE) between feature representations 
    extracted from the 4th convolutional layer (Conv4_4) of a pre-trained 
    VGG19 network. This loss encourages the model to preserve semantic 
    content and structural textures rather than just pixel-level alignment.
    """

    def __init__(self) -> None:
        super(VGGLoss, self).__init__()
        
        # Extract features up to the relu4_4 layer (index 35)
        # This layer is the research standard for perceptual super-resolution
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg.children())[:36]).eval()
        
        # Freeze VGG parameters to ensure deterministic feature extraction
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()  # L1 for sharper perceptual gradients (matches FeatureExtractor)

        # Register ImageNet normalization constants as buffers.
        # This ensures they are moved to the correct GPU/device automatically 
        # but are not included in the model's trainable parameters.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the perceptual loss between reconstructed and target images.

        Parameters
        ----------
        x : torch.Tensor
            Predicted/Super-resolved image tensor in range [-1, 1].
        y : torch.Tensor
            Ground-truth high-resolution image tensor in range [-1, 1].

        Returns
        -------
        torch.Tensor
            Scalar perceptual loss value.
        """
        # Use modern PyTorch API to force FP32 and prevent activation NaNs
        with torch.amp.autocast('cuda', enabled=False):
            # 1. Rescale inputs from [-1, 1] to [0, 1] for VGG compatibility
            x = (x.float() + 1.0) / 2.0
            y = (y.float() + 1.0) / 2.0
    
            # 2. Standardize images using ImageNet statistics
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
    
            # 3. Extract semantic feature maps
            # Use gradient checkpointing if handling high-res image tensors to save VRAM
            if x.shape[-1] >= 256: 
                x_features = checkpoint(self.vgg, x, use_reentrant=False)
                y_features = checkpoint(self.vgg, y, use_reentrant=False)
            else:
                x_features = self.vgg(x)
                y_features = self.vgg(y)
    
            # 4. Normalize features to unit variance before computing loss
            # This prevents high-magnitude feature channels from dominating the gradient
            x_features = x_features / (x_features.norm(dim=1, keepdim=True) + 1e-8)
            y_features = y_features / (y_features.norm(dim=1, keepdim=True) + 1e-8)
    
            # 5. Compute L1 distance in the normalized feature manifold
            return self.criterion(x_features, y_features)

    def train(self, mode: bool = True):
        """
        Override train mode to ensure VGG always remains in evaluation mode.
        This prevents BatchNorm/Dropout layers from behaving unpredictably.
        """
        super().train(False)
        return self