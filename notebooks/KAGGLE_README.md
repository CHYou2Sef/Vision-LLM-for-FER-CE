# Kaggle Notebooks for FER-CE Project

This folder contains Kaggle-optimized versions of all notebooks for the `raf-ce-2026` dataset.

## Dataset Setup on Kaggle

1. **Add Dataset**: In your Kaggle notebook, click "Add Data" → Search for `raf-ce-2026` → Add to notebook
2. **Enable GPU**: Settings → Accelerator → GPU T4 Tesla

## Notebooks

### 1. ResNet50 Baseline
**File**: `kaggle_01_baseline_resnet.ipynb`
- Standard CNN baseline
- ~20 epochs training
- Outputs: Model weights + confusion matrix

### 2. ViT Baseline
**File**: `kaggle_02_baseline_vit.ipynb`
- Vision Transformer approach
- Data analysis visualizations
- Training curves + detailed metrics

### 3. BLIP-2 Vision-LLM
**File**: `kaggle_03_vision_llm.ipynb`
- Multimodal emotion explanation
- Zero-shot inference examples
- LoRA fine-tuning ready

## Key Differences from Local Notebooks

- **Paths**: Hardcoded to `/kaggle/input/raf-ce-2026`
- **Dataset Class**: Inline (no separate `src/dataset.py`)
- **Outputs**: Saved to `/kaggle/working/`
- **Dependencies**: Auto-installed in first cell

## Usage

1. Upload notebook to Kaggle
2. Add `raf-ce-2026` dataset
3. Enable GPU
4. Run all cells

All outputs (models, plots) will be in `/kaggle/working/` and can be downloaded after execution.
