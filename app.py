import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# Page title
st.title("Skin Cancer Detection (Melanoma vs Benign)")
st.write("Upload a dermoscopic image to get a prediction.")
st.warning("Educational project only. This is not a medical device and must not be used for diagnosis.")

# Device
device = torch.device("cpu")

# Load model (same architecture as training, see model.py)
model = get_model(pretrained=False)
model.load_state_dict(torch.load("melanoma_model.pth", map_location=device))
model.eval()

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_names = ["Benign", "Melanoma"]

    st.subheader("Prediction")
    st.write(f"**Class:** {class_names[predicted.item()]}")
    st.write(f"**Confidence:** {confidence.item()*100:.2f}%")
