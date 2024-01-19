import torch
import uvicorn
import hydra
import albumentations as A
import json
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse
import sys
import os
import subprocess
import re
from datetime import datetime
from pathlib import Path
from google.cloud import storage
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.make_dataset import ImgTransformer
from src.models.model import EfficientNetV2Model

app = FastAPI()

def download_latest_model(bucket_name, prefix):
    # initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # list all blobs in the specified gcloud
    blobs = list(bucket.list_blobs(prefix=prefix))

    # filter for .ckpt files and sort by creation time
    ckpt_files = [blob for blob in blobs if blob.name.endswith('.ckpt')]
    latest_blob = max(ckpt_files, key=lambda b: b.time_created)
    print(latest_blob)
    if latest_blob:
        model_path = Path('latest_model.ckpt')
        latest_blob.download_to_filename(model_path)
        return model_path
    else:
        raise Exception("No .ckpt files found in the specified bucket and prefix")

# bucket name and prefix
bucket_name = 'data_tensors'
prefix = '' 

# download the latest model
latest_model_path = download_latest_model(bucket_name, prefix)
model = EfficientNetV2Model.load_from_checkpoint(latest_model_path)


# redirect on root
@app.get("/")
async def root():
    return RedirectResponse(url="/predict")


@app.get("/predict")
async def predict_get():
    return "Send a POST request to /predict with an image file to get a prediction"


@app.post("/predict")
async def predict_post(data: UploadFile):
    # load uploaded image using PIL
    image = Image.open(data.file)
    transformer = ImgTransformer(pipeline=A.Compose([A.Resize(width=90, height=90)]))
    tensor = transformer(image).unsqueeze(0)

    label_to_emotion = {'0': 'Ahegao', '1': 'Angry', '2': 'Happy', '3': 'Neutral', '4': 'Sad', '5': 'Surprise'}

    model.eval()
    logits = model(tensor)
    pred_prob = torch.nn.functional.softmax(logits)
    prediction = torch.argmax(pred_prob)
    prediction = prediction.item()
    prediction_label = str(prediction)
    prediction_class = label_to_emotion[prediction_label]

    return {"Image shape": tensor.shape, "Predicted label": {prediction_label}, "Predicted class": {prediction_class}}


#@hydra.main(config_path="../../conf", config_name="config.yaml", version_base="1.3.2")
def run():
    """
    web_conf, model_conf = (
        config.web.fastapi,
        config.model.default_model
    )

    global model
    model = model_conf.paths.model_checkpoint
    """

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
