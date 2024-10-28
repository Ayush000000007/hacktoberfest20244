import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Paths and device setup
image_path = '/content/drive/My Drive/DATASET/test/1/sjchoi86-HRF-398.png'
model_path = '/content/drive/My Drive/final_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the model
def load_model(model_path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.classifier[1].in_features, 1),
        torch.nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image.unsqueeze(0)  # Add batch dimension

# Make prediction
def predict(image_path, model):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        prediction = output.item()  # Sigmoid output
    return "Positive" if prediction > 0.5 else "Negative", prediction

# Load model and predict
model = load_model(model_path)
label, confidence = predict(image_path, model)
print(f"Prediction: {label}, Confidence: {confidence:.4f}")
