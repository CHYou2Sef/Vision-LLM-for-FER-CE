# FER-CE: Face Emotion Recognition with Vision-LLM

This project explores compound emotion recognition using the RAF-CE dataset. It benchmarks a classic **ResNet50** baseline against a multimodal **BLIP-2** Vision-LLM.

## Project Structure

- `notebooks/`:
    - `01_baseline_vision.ipynb`: Training and evaluation of ResNet50.
    - `02_vision_llm.ipynb`: Fine-tuning BLIP-2 with LoRA for explanations.
- `src/`:
    - `dataset.py`: Common RAF-CE dataset loader.
- `outputs/`: Saved models, plots, and classification reports.
- `Dockerfile`: Container environment for reproducibility.

## How to Run

### Local Setup
1. Install dependencies:
   ```bash
   pip install torch torchvision transformers peft pandas pillow scikit-learn matplotlib seaborn
   ```
2. Run notebooks in the `notebooks/` folder.

### Docker Setup
1. Build the container:
   ```bash
   docker build -t fer-ce-project .
   ```
2. Run the container:
   ```bash
   docker run -p 8888:8888 fer-ce-project
   ```
3. Open the link displayed in the terminal to access Jupyter Lab.

## Models

- **Baseline**: ResNet50 pre-trained on ImageNet, fine-tuned on 14 compound emotion classes.
- **Vision-LLM**: BLIP-2 (Salesforce/blip2-opt-2.7b) fine-tuned using LoRA to generate both the emotion label and a textual justification based on facial cues.
