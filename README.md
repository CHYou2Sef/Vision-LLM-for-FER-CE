# Facial Emotion Recognition of Compound Expressions (FER-CE) via Vision-LLM 🎭

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</p>

## 📌 Context and Objectives
This project was developed as part of the **Data Mining & Artificial Intelligence** course at ENICARTHAGE. 

Traditional facial emotion recognition (FER) models often fail in real-world scenarios where humans express mixed emotions. The goal of this project is to go beyond the 7 basic emotions and classify **14 compound emotions** (e.g., *Happily Surprised*, *Fearfully Disgusted*) using the **RAF-CE dataset**. 

We explored multimodal foundation models (Vision-Language) to leverage their semantic understanding.

## 🚀 Technical Stack
* **Frameworks:** PyTorch, Hugging Face Transformers.
* **Optimization:** PEFT (Parameter-Efficient Fine-Tuning) for LoRA implementation, Accelerate & BitsAndBytes (4-bit/8-bit quantization).
* **Data Processing & Visualization:** OpenCV, Scikit-learn, Matplotlib, Seaborn.
* **Infrastructure:** Google Colab (NVIDIA T4 GPU).

## 🧠 Methodology: Overcoming the "Small Data" Wall
The RAF-CE dataset presents a massive challenge: extreme class imbalance and very little data (~900 training images). Training a large model from scratch would lead to immediate overfitting.

To solve this, our pipeline includes:
1. **Model Shift:** We transitioned from Generative (BLIP-2) to Contrastive (CLIP) models for better stability and task alignment.
2. **LoRA Fine-Tuning:** We fine-tuned CLIP using **LoRA (r=32)**, training only the attention matrices while keeping the core knowledge intact.
3. **Data Optimization:** Implemented *Weighted Random Sampling* for rare classes, aggressive Data Augmentation (Color Jitter, Gaussian Blur), and *Prompt Ensembling*.

## 📊 Results & Critical Analysis
While pre-trained models like ViT or ResNet achieve higher raw accuracy on large datasets, fine-tuning CLIP on such a constrained dataset ("Small Data") achieved a **55% Accuracy** and **0.34 F1-Score**. 

**Key Learnings:**
* **Interpretability:** Vision-LLMs provide rich semantic understanding and allow us to visualize attention zones (eyes, mouth).
* **Limitations:** The model is heavily dependent on prompts and struggles to converge fully with less than 1000 images. However, the approach proves the feasibility of adapting foundation models with LoRA under heavy constraints.

## 👥 Contributors
* Badii BAZAOUI 
* Omar RJAB 
* Yassine BEN JDIDA 
* Yousef CHEBL
*(Group Project - ENICARTHAGE 2026)*
