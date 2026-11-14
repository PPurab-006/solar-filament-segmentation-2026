"""
Solar Filament Segmentation Package (solarseg).

Provides stage 1 pipeline components for the Solar Filament Segmentation Challenge 2026:
- Data parsing and grouping by observation date
- Solar disk limb detection and limb-darkening correction preprocessing
- Full-resolution crop dataset with soft multi-annotator labels
- ResNet34 U-Net baseline architecture
- Panoptic Quality (PQ) metric evaluator
- Hann-window blended sliding-window inference
- Post-processing and COCO RLE submission generation
"""

__version__ = "0.1.0"
