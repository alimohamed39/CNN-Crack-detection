from model import get_model
import numpy as np
import os
import torch
from torchvision import datasets,transforms
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

model = get_model()
model.eval()


transform = transforms.Compose([transforms.Resize((224,224))

                                   ,transforms.ToTensor()
                                   , transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])



main_path = r"C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\Surface Crack detection proj"

img_path =os.path.join(main_path,"non_cracked_granite.jpg") # you can put any image name as a paramter her
image = Image.open(img_path)
image = transform(image).unsqueeze(0)

model.load_state_dict(torch.load(r'C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\Surface Crack detection proj\best_model.pth'))
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    if predicted.item() == 1:
        cl = "cracked"
    else:
        cl = "not Cracked"


    print("The object is",cl)
    print(predicted.item())


