import uvicorn
import torch
import hydra
import albumentations as A

from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse

from src.data.make_dataset import ImgTransformer

app = FastAPI()

model_path = "models/checkpoints/model_all_data_epoch_5_lr_0.0001.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # transform image
    transformer = ImgTransformer(pipeline=A.Compose([A.Resize(width=90, height=90)]))

    # convert image to tensor and add batch dimension [1, 3, 90, 90]
    tensor = transformer(image).unsqueeze(0)

    # model = EfficientNetV2Model()
    # model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # preds = model(tensor)

    return {"image tensor shape": tensor.shape}


@hydra.main(config_path="../..", config_name="config", version_base="1.3.2")
def run(config):
    global checkpoint
    checkpoint = config.paths.model_checkpoint

    uvicorn.run(app, host=config.web.host, port=config.web.port)


if __name__ == "__main__":
    run()
