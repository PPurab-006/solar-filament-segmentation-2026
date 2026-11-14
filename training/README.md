# Model Training Pipeline (R1 ConvNeXt-Tiny)

This directory contains the standalone GPU training script used to generate the final model weights (`weights/r1_convnext_tiny_ema_weights.pt`).

## Architecture & Configuration
- **Model**: Segmentation Models PyTorch (SMP) `Unet`
- **Encoder**: `tu-convnext_tiny` initialized from ImageNet weights
- **Input Channels**: 3 (Primary normalized H-alpha solar disk image + 2 spatial context channels)
- **Loss**: Combined Dice Loss (squared denominator, $\epsilon=1.0$) + Binary Cross Entropy (BCE) with equal 1.0 weights
- **Optimizer**: AdamW ($\text{lr}=3\times 10^{-4}$, weight decay $10^{-2}$)
- **LR Schedule**: Cosine Annealing over 20 epochs down to $\text{lr}_{\min}=10^{-6}$
- **Batch Size**: 8
- **EMA**: Exponential Moving Average with decay $\beta=0.999$ tracked across all epochs
- **Target Precision**: Final weights stored in half precision (`float16`), totaling ~61 MB

## Training Execution
To train on a Kaggle GPU (P100 / T4) or local CUDA machine:

```bash
python training/train_r1_gpu.py \
    --data-dir /path/to/MAGFiLO_1.0_Kaggle_2026/train \
    --epochs 20 \
    --batch-size 8 \
    --lr 3e-4 \
    --out-weights weights/r1_convnext_tiny_ema_weights.pt
```

The script trains for 20 epochs, calculates validation metrics at each epoch, updates the EMA model, and saves the final half-precision EMA weights and metadata JSON.
