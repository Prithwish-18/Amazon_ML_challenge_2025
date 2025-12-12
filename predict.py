"""
Simple inference wrapper: given image path and catalog_content, predict by image model (PyTorch) and text model (joblib) and average.
Usage:
 python src/predict.py --image data/images/123.jpg --catalog "Item Name: ..." --image-model model/image_model.pth --text-model model/text_model.joblib
"""
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np

def load_image_model(path, device):
    m = models.resnet18(pretrained=False)
    n = m.fc.in_features
    import torch.nn as nn
    m.fc = nn.Linear(n,1)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    m.eval()
    return m

def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(x).cpu().numpy().ravel()[0]
    return float(p)

def predict_text(model, text):
    p = model.predict([text])[0]
    return float(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--image-model", required=True)
    parser.add_argument("--text-model", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model = load_image_model(args.image_model, device)
    text_model = joblib.load(args.text_model)
    p_img = predict_image(img_model, args.image, device)
    p_text = predict_text(text_model, args.catalog)
    # simple average ensemble
    final = (p_img + p_text) / 2.0
    print(f"image_pred: {p_img:.4f}, text_pred: {p_text:.4f}, final: {final:.4f}")