import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
from typing import List

app = FastAPI()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
model_path = '../weights/model_mobilenet_celeb_quantized.pth'
model = models.mobilenet_v2(pretrained=False).to(device)
model.classifier[1] = nn.Linear(in_features=1280, out_features=2, bias=True)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

list_of_clothes = ['blouse',
                   'trousers',
                   'turtleneck',
                   'jumper',
                   'jeans',
                   'jacket',
                   'vest',
                   'cardigan',
                   'suit',
                   'swimsuit',
                   'blazer',
                   'dress',
                   'polo',
                   'pullover',
                   'shirt',
                   'sweater',
                   'sweatshirt',
                   'hoody',
                   'top',
                   't-shirt',
                   'shorts',
                   'skirt']

model_path_classification = '../weights/model_resnet18_quantized_clothes_class.pth'
model_classification_clothes = models.resnet18(pretrained=False).to(device)
model_classification_clothes.fc = nn.Linear(in_features=512, out_features=len(list_of_clothes), bias=True)
model_classification_clothes.load_state_dict(torch.load(model_path_classification, map_location=device))
model_classification_clothes.eval()


def load_img(image_data: bytes, img_transforms: transforms = data_transforms) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_data))
    tensor_image = img_transforms(image)
    return tensor_image.unsqueeze(0)


def vectorize_image(image_data: bytes) -> np.array:
    tensor_image = load_img(image_data, data_transforms)
    with torch.no_grad():
        tensor_image = tensor_image
        res = model(tensor_image)
        _, preds = torch.max(res, 1)
        sex = 'male' if preds == 0 else 'female'
    return sex


def vectorize_image_clothes(image_data: bytes) -> np.array:
    tensor_image = load_img(image_data, data_transforms)
    with torch.no_grad():
        tensor_image = tensor_image
        res = model_classification_clothes(tensor_image)
        _, preds = torch.max(res, 1)
        clothes = list_of_clothes[preds]
    return clothes


class Images(BaseModel):
    image: bytes

@app.post('/classification_sex')
def vectorize(image_data: UploadFile = File(...)):
    embedding = vectorize_image(image_data.file.read())
    result = {
        image_data.filename: embedding
    }
    return result


@app.post('/classification_clothes')
def vectorize(image_data: UploadFile = File(...)):
    embedding = vectorize_image_clothes(image_data.file.read())
    result = {
        image_data.filename: embedding
    }
    return result


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

