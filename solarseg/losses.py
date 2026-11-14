"""
Loss functions and extensible loss registry for solar filament segmentation.

WHY THIS EXISTS:
Filament segmentation requires balancing pixel-level accuracy with region-level overlap:
1. Binary Cross-Entropy (BCE) provides smooth, convex pixel-wise gradients for both
   confident and ambiguous (soft-labeled) pixels.
2. Soft Dice Loss directly maximizes region overlap, combating the extreme class imbalance
   where filament pixels represent <1% of the field.
A loss registry allows future stages (e.g. clDice, Skeleton Recall, TRACE topology loss)
to slot in seamlessly via decorator registration without touching training code.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Global loss registry mapping names to factory functions
LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_loss(name: str) -> Callable:
    """
    Decorator to register a loss class or factory in the global registry.

    WHY THIS EXISTS:
    Allows modular extension so future topology losses (e.g. clDice, Skeleton Recall)
    can be plugged in via @register_loss('cldice') without refactoring the training loop.
    """
    def decorator(cls_or_fn: Callable) -> Callable:
        LOSS_REGISTRY[name.lower()] = cls_or_fn
        return cls_or_fn

    return decorator


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Retrieve and instantiate a loss function from the registry.

    Args:
        name: Registered name (e.g., 'bce_dice', 'bce', 'dice').
        **kwargs: Arguments forwarded to the loss constructor.
    """
    name_clean = name.lower()
    if name_clean not in LOSS_REGISTRY:
        raise KeyError(
            f"Loss '{name}' not found in registry. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name_clean](**kwargs)


@register_loss("skeleton_recall")
class SkeletonRecallLoss(nn.Module):
    """
    Skeleton Recall Loss (Part E2):
    Penalizes false negatives on thin filament topological skeletons.
    Calculates soft recall: sum(p * y_skel) / (sum(y_skel) + eps)
    Loss = 1.0 - recall
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        ground_truth = targets.sum(dim=(-2, -1))
        recall = (intersection + self.smooth) / (ground_truth + self.smooth)
        return 1.0 - recall.mean()


@register_loss("dice")
class SoftDiceLoss(nn.Module):
    """
    Continuous Soft Dice Loss supporting soft multi-annotator targets.

    WHY THIS EXISTS:
    Targets are continuous [0, 1] agreement fractions. Standard hard Dice fails on soft targets.
    This formulation computes the soft intersection and union directly on sigmoid probabilities.
    """

    def __init__(self, smooth: float = 1e-6, square_denominator: bool = True):
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw unactivated network outputs.
            targets: (B, 1, H, W) continuous target masks in [0.0, 1.0].

        Returns:
            Scalar loss tensor.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions per batch item
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)

        if self.square_denominator:
            cardinality = torch.sum(probs_flat**2, dim=1) + torch.sum(targets_flat**2, dim=1)
        else:
            cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice_score)


@register_loss("bce")
class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with logits for soft labels.

    WHY THIS EXISTS:
    PyTorch's BCEWithLogitsLoss natively handles continuous targets in [0, 1],
    computing cross-entropy relative to annotator confidence fractions.
    """

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        pos_wt_tensor = torch.tensor([pos_weight]) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


@register_loss("bce_dice")
class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Soft Dice Loss.

    WHY THIS EXISTS:
    Default loss for Stage 1 baseline. Combining pixel-wise BCE and region-wise Dice
    stabilizes training and accelerates boundary convergence.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge loss for flattened 1D tensors."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


@register_loss("lovasz_hinge")
class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss for binary segmentation (Berman et al., CVPR 2018).
    Directly optimizes the continuous surrogate of the Jaccard index / IoU.
    Operates on thresholded binary targets (targets > 0.5).
    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bin_targets = (targets > 0.5).float()
        if self.per_image:
            losses = [
                _lovasz_hinge_flat(logits[i].view(-1), bin_targets[i].view(-1))
                for i in range(logits.size(0))
            ]
            return torch.stack(losses).mean()
        return _lovasz_hinge_flat(logits.view(-1), bin_targets.view(-1))


@register_loss("bce_dice_lovasz")
class BCEDiceLovaszLoss(nn.Module):
    """
    Combined BCE + Soft Dice + Lovasz-Hinge Loss (Stage 2b Part B2 recipe).
    Loss = BCE + Soft Dice + 0.3 * Lovasz-Hinge.
    BCE and Soft Dice train directly on continuous soft-consensus targets.
    Lovasz-Hinge operates on binarized consensus targets (targets > 0.5).
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        lovasz_weight: float = 0.3,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_bce = self.bce(logits, targets)
        l_dice = self.dice(logits, targets)
        l_lov = self.lovasz(logits, targets)
        return (
            self.bce_weight * l_bce
            + self.dice_weight * l_dice
            + self.lovasz_weight * l_lov
        )


@register_loss("barb_weighted_bce_dice")
class BarbWeightedBCEDiceLoss(nn.Module):
    """
    Task 5c: Barb-Weighted BCE + Soft Dice Loss.
    Applies per-pixel weight map w = 1 + beta * (mask AND NOT dilate(spine, r))
    to the BCE and Soft Dice terms only.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model predictions.
            targets: (B, 1, H, W) target consensus mask.
            weight_map: Optional (B, 1, H, W) float32 weights. If None, uniform 1.0.
        """
        if weight_map is None:
            weight_map = torch.ones_like(targets)

        # 1. Weighted BCE
        bce_unreduced = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted_bce = (bce_unreduced * weight_map).sum() / weight_map.sum().clamp(min=1.0)

        # 2. Weighted Soft Dice
        probs = torch.sigmoid(logits)
        intersection = (weight_map * probs * targets).sum(dim=(-2, -1))
        cardinality = (weight_map * (probs**2 + targets**2)).sum(dim=(-2, -1))
        weighted_dice = 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth)).mean()

        return self.bce_weight * weighted_bce + self.dice_weight * weighted_dice


@register_loss("multitask_auxiliary")
class MultiTaskAuxiliaryLoss(nn.Module):
    """
    Multi-task loss combining Primary Mask Loss with optional Centerline (Skeleton/Spine)
    and Boundary/Distance-Transform heads.
    """

    def __init__(
        self,
        mask_loss: Optional[nn.Module] = None,
        centerline_loss: Optional[nn.Module] = None,
        boundary_loss: Optional[nn.Module] = None,
        weight_mask: float = 1.0,
        weight_centerline: float = 0.5,
        weight_boundary: float = 0.5,
    ):
        super().__init__()
        self.mask_loss = mask_loss or BCEDiceLovaszLoss()
        self.centerline_loss = centerline_loss or SkeletonRecallLoss()
        self.boundary_loss = boundary_loss or nn.MSELoss()
        self.weight_mask = weight_mask
        self.weight_centerline = weight_centerline
        self.weight_boundary = weight_boundary

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weight_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        mask_logits = preds["mask"]
        mask_target = targets["mask"]

        if isinstance(self.mask_loss, BarbWeightedBCEDiceLoss):
            l_mask = self.mask_loss(mask_logits, mask_target, weight_map=weight_map)
        else:
            l_mask = self.mask_loss(mask_logits, mask_target)

        loss_dict["mask_loss"] = l_mask
        total_loss = self.weight_mask * l_mask

        if "centerline" in preds and "centerline" in targets:
            l_cent = self.centerline_loss(preds["centerline"], targets["centerline"])
            loss_dict["centerline_loss"] = l_cent
            total_loss = total_loss + self.weight_centerline * l_cent

        if "boundary" in preds and "boundary" in targets:
            l_bound = self.boundary_loss(torch.sigmoid(preds["boundary"]), targets["boundary"])
            loss_dict["boundary_loss"] = l_bound
            total_loss = total_loss + self.weight_boundary * l_bound

        loss_dict["total_loss"] = total_loss
        return loss_dict
