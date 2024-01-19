import uvicorn
import torch
import hydra
import albumentations as A

from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse
import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.make_dataset import ImgTransformer
from src.models.model import EfficientNetV2Model

app = FastAPI()

model_path = f"{project_root}/src/models/checkpoints/model.ckpt"
model = EfficientNetV2Model.load_from_checkpoint(model_path)

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
