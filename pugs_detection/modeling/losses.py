import segmentation_models_pytorch as smp
import torch.nn as nn

class JaccardFocalLoss(nn.Module):
    
    def __init__(self, mode="binary", jaccard_weight=0.5,focal_weight=0.5, ignore_index=None, normalized=True):
        super().__init__()
        self.jaccard =smp.losses.JaccardLoss(mode="binary")
        self.focal = smp.losses.FocalLoss(mode=mode, ignore_index=ignore_index, normalized=normalized)
        self.jaccard_weight = jaccard_weight
        self.focal_weight = focal_weight


    def forward(self, y_pred, y_true):
        jaccard_loss = self.jaccard(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        return (self.jaccard_weight * jaccard_loss) + (self.focal_weight * focal_loss)

