"""
Simple transfer-learning training script using PyTorch.
Usage:
 python src/train_image_model.py --images data/images --csv data/reduced_train.csv --model-out model/image_model.pth --epochs 5
Notes:
 - Expects images named <sample_id>.* matching CSV sample_id.
 - CSV must contain 'sample_id' and 'price'.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from src.utils import setup_logging, makedirs
import logging

setup_logging()
logger = logging.getLogger(__name__)

class SimpleImageDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, id_col="sample_id", target_col="price"):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.id_col = id_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row[self.id_col]
        # try common extensions
        candidates = list(self.images_dir.glob(f"{sid}.*"))
        if len(candidates) == 0:
            # return a zero image if missing
            img = Image.new("RGB", (224,224), color=(127,127,127))
        else:
            img = Image.open(candidates[0]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return img, y

def make_model():
    m = models.resnet18(pretrained=True)
    n = m.fc.in_features
    m.fc = nn.Linear(n, 1)
    return m

def train(args):
    df = pd.read_csv(args.csv)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    dataset = SimpleImageDataset(df, args.images, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / len(dataset)
        logger.info("Epoch %d loss: %.4f", epoch+1, epoch_loss)
    makedirs(Path(args.model_out).parent)
    torch.save(model.state_dict(), args.model_out)
    logger.info("Saved model to %s", args.model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)