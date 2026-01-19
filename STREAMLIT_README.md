# FER-CE Streamlit Demo

## Installation

```bash
pip install streamlit torch torchvision transformers pillow opencv-python matplotlib numpy
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

- **Multi-Model Support**: Choose between ResNet50, ViT-B/16, or BLIP-2
- **Real-time Prediction**: Upload an image and get instant emotion classification
- **Explainability**: BLIP-2 provides textual explanations
- **Probability Distribution**: Visualize confidence across all 14 emotions
- **User-Friendly Interface**: Clean, intuitive design

## Usage

1. Select a model from the sidebar
2. Upload a facial image (JPG, PNG)
3. View the predicted emotion and confidence
4. For BLIP-2: Read the AI-generated explanation

## Model Comparison

| Model | Speed | Accuracy | Explainability |
|-------|-------|----------|----------------|
| ResNet50 | ⚡⚡⚡ | ⭐⭐ | ❌ |
| ViT-B/16 | ⚡⚡ | ⭐⭐⭐ | ❌ |
| BLIP-2 | ⚡ | ⭐⭐ | ✅ |

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for BLIP-2)
- Model weights in `outputs/` folder

## Troubleshooting

**Model weights not found**:
- Ensure you've trained the models first
- Check that `.pth` files are in the correct `outputs/` subdirectories

**Out of memory**:
- Use CPU mode (automatic fallback)
- Reduce batch size or use ResNet50 instead of BLIP-2

**Slow inference**:
- Enable GPU
- Use ResNet50 for fastest results
