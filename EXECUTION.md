# Execution notes (short)

1) Install requirements
   pip install -r requirements.txt

2) Prepare /data:
   - Put train.csv (original) in data/train.csv
   - Optionally sample down (20000 rows):
     python src/data_prep.py --input data/train.csv --out-csv data/reduced_train.csv --sample 20000

3) Download images:
   python src/download_images.py --csv data/reduced_train.csv --out-dir data/images --limit 20000

4) Train:
   - Image model (GPU recommended):
     python src/train_image_model.py --images data/images --csv data/reduced_train.csv --model-out model/image_model.pth --epochs 5
   - Text baseline:
     python src/train_text_model.py --csv data/reduced_train.csv --model-out model/text_model.joblib

5) Predict:
   python src/predict.py --image data/images/33127.jpg --catalog "Item Name: ..." --image-model model/image_model.pth --text-model model/text_model.joblib

Tips
----
- Use larger backbones and better augmentation for improved image performance.
- Consider feature extraction from catalog_content (value/unit) as additional numeric features.