# Amazon_ML_challenge_2025— Solution (Prithwish)

Overview
--------
This repository contains a reproducible solution for the Amazon ML Challenge 2025. It includes scripts for data preparation, downloading images, training an image model (transfer learning with PyTorch), a baseline text model, and inference utilities.

Repository layout
-----------------
- data/                 : Place dataset CSVs here (not included)
- model/                : Trained model weights (not included)
- notebooks/            : (optional) exploratory notebooks
- src/                  : Project source code
- docs/                 : Execution notes and quick reference
- requirements.txt
- README.md

Quick start (local)
-------------------
1. Clone the repo:
   git clone https://github.com/<OWNER>/Amazon_ML_challenge_2025_prithwish.git
   cd Amazon_ML_challenge_2025_prithwish

2. Create a Python virtual environment and install deps:
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

3. Prepare data
   - Put your training CSV at data/train.csv (train CSV originally: student_resource/dataset/train.csv).
   - Example:
     python src/data_prep.py --input data/train.csv --out-csv data/reduced_train.csv --sample 20000

   - Download images (saves into data/images):
     python src/download_images.py --csv data/reduced_train.csv --out-dir data/images --id-col sample_id --url-col image_link

4. Train image model (transfer learning, requires GPU recommended)
   python src/train_image_model.py \
       --images data/images \
       --csv data/reduced_train.csv \
       --model-out model/image_model.pth \
       --epochs 5 \
       --batch-size 32 \
       --lr 1e-4

5. (Optional) Train text model
   python src/train_text_model.py --csv data/reduced_train.csv --model-out model/text_model.joblib

6. Inference
   python src/predict.py --image-path data/images/12345.jpg --catalog "Item Name: ..." --model-image model/image_model.pth --model-text model/text_model.joblib

Notes
-----
- The image model uses a torchvision pretrained ResNet18 and a small head for regression. For competitive performance, replace with stronger backbones and fine-tune more epochs.
- The dataset CSV contains columns: sample_id, catalog_content, image_link, price. Adjust CLI args if your column names differ.
- GPU is strongly recommended when training the image model.

Reproducibility
---------------
- Use the provided requirements.txt; pinning exact versions is recommended for full reproducibility.
- Save model weights to model/ and log hyperparameters.

License
-------
MIT
