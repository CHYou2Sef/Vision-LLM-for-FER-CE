import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os

# Page config
st.set_page_config(
    page_title="FER-CE: Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# Title
st.title("🎭 FER-CE: Compound Emotion Recognition")
st.markdown("**Vision-LLM for Facial Emotion Recognition and Compound Expressions**")
st.markdown("---")

# Emotion mapping
EMO_MAP = {
    0: "Happily surprised", 1: "Happily disgusted", 2: "Sadly fearful", 3: "Sadly angry",
    4: "Sadly surprised", 5: "Sadly disgusted", 6: "Fearfully angry", 7: "Fearfully surprised",
    8: "Fearfully disgusted", 9: "Angrily surprised", 10: "Angrily disgusted",
    11: "Disgustedly surprised", 12: "Happily fearful", 13: "Happily sad"
}

# Sidebar - Model Selection
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["ResNet50 (Fast)", "ViT-B/16 (Accurate)", "BLIP-2 (Explainable)"]
)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.info(f"🖥️ Device: {DEVICE}")

# Cache models
@st.cache_resource
def load_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 14)
    model.load_state_dict(torch.load('outputs/baseline/resnet50_ferce_baseline.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_vit():
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, 14)
    # Load weights if available
    try:
        model.load_state_dict(torch.load('outputs/baseline/ViT/vit_baseline.pth', map_location=DEVICE))
    except:
        st.warning("ViT weights not found, using pre-trained only")
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    return processor, model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Prediction functions
def predict_resnet(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return pred.item(), probs[0].cpu().numpy()

def predict_vit(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return pred.item(), probs[0].cpu().numpy()

def predict_blip2(processor, model, image):
    prompt = "Question: Describe the emotional state of this person and explain which facial cues contribute to it. Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    explanation = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Extract emotion from explanation
    predicted_label = 0
    for emo_id, emo_name in EMO_MAP.items():
        if emo_name.lower() in explanation.lower():
            predicted_label = emo_id
            break
    
    return predicted_label, explanation

# Main app
st.header("📤 Upload Image")

uploaded_file = st.file_uploader("Choose a facial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🔮 Prediction")
        
        with st.spinner('Analyzing emotion...'):
            if "ResNet50" in model_choice:
                model = load_resnet50()
                image_tensor = preprocess_image(image)
                pred_id, probs = predict_resnet(model, image_tensor)
                
                st.success(f"**Predicted Emotion**: {EMO_MAP[pred_id]}")
                st.metric("Confidence", f"{probs[pred_id]*100:.2f}%")
                
                # Top 3 predictions
                st.write("**Top 3 Predictions:**")
                top3_idx = np.argsort(probs)[-3:][::-1]
                for idx in top3_idx:
                    st.write(f"- {EMO_MAP[idx]}: {probs[idx]*100:.1f}%")
            
            elif "ViT" in model_choice:
                model = load_vit()
                image_tensor = preprocess_image(image)
                pred_id, probs = predict_vit(model, image_tensor)
                
                st.success(f"**Predicted Emotion**: {EMO_MAP[pred_id]}")
                st.metric("Confidence", f"{probs[pred_id]*100:.2f}%")
                
                # Top 3 predictions
                st.write("**Top 3 Predictions:**")
                top3_idx = np.argsort(probs)[-3:][::-1]
                for idx in top3_idx:
                    st.write(f"- {EMO_MAP[idx]}: {probs[idx]*100:.1f}%")
            
            elif "BLIP-2" in model_choice:
                processor, model = load_blip2()
                pred_id, explanation = predict_blip2(processor, model, image)
                
                st.success(f"**Predicted Emotion**: {EMO_MAP[pred_id]}")
                st.info(f"**Explanation**: {explanation}")
    
    # Probability distribution
    if "BLIP-2" not in model_choice:
        st.subheader("📊 Probability Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        emotions = [EMO_MAP[i] for i in range(14)]
        ax.barh(emotions, probs, color='skyblue')
        ax.set_xlabel('Probability')
        ax.set_title('Emotion Probability Distribution')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)

# Sidebar - Info
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**Models**:
- **ResNet50**: Fast CNN baseline
- **ViT-B/16**: High accuracy Transformer
- **BLIP-2**: Explainable Vision-LLM

**Dataset**: RAF-CE (14 compound emotions)

**Author**: [Your Name]  
**Course**: Data Mining & AI
""")

# Footer
st.markdown("---")
st.markdown("🎓 **ENICar - Data Mining & Intelligence Artificielle**")
