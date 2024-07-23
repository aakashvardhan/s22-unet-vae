import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    # Multi class Dice Loss for Image Segmentation using softmax
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, y_pred, y_true):
        # Ensure y_true is of type torch.long for F.one_hot
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        # Check the shape of y_true and remove extra dimension if present
        if y_true.dim() == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)  # Removes the channel dimension

        # Apply softmax to y_pred if required
        prob = y_pred
        if self.config["softmax_dim"] is not None:
            prob = nn.Softmax(dim=self.config["softmax_dim"])(y_pred)

        # Convert y_true to one-hot encoding and rearrange dimensions
        y_true = F.one_hot(y_true, num_classes=3)
        y_true = y_true.permute(0, 3, 1, 2).float()

        # Calculate dice loss
        numerator = 2 * torch.sum(prob * y_true, dim=(2, 3))
        denominator = torch.sum(prob.pow(2) + y_true.pow(2), dim=(2, 3))
        dice_loss = 1 - (numerator + 1) / (denominator + 1)

        return dice_loss.mean()
