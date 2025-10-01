from model import get_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

transform = transforms.Compose([transforms.Resize((224,224))
                                   ,transforms.ToTensor()
                                   , transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(root=r"C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\Surface Crack detection proj\train",transform=transform)
val_data = datasets.ImageFolder(root=r"C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\Surface Crack detection proj\val",transform=transform)




def train(model,batch_size,optim,epochs,lr,loss_f=torch.nn.CrossEntropyLoss(),show_progress=True):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    epochs_range = range(epochs)
    optim = optim(model.parameters(),lr=lr)

    best_loss = float("inf")
    for epoch in (tqdm(range(epochs), desc="Training Epochs") if show_progress else epochs_range):
        model.train()
        running_loss = 0.0
        for image,label in train_loader:
            output = model(image)

            loss = loss_f(output, label)
            loss.backward()
            optim.step()
            optim.zero_grad()


            running_loss += loss.item()
        average_train_loss = running_loss / len(train_loader)

        with torch.no_grad():
            val_loss = 0.0
            model.eval()

            for image,label in val_loader:
                output = model(image)
                loss = loss_f(output, label)
                val_loss += loss.item()

            average_val_loss = val_loss / len(val_loader)

            if average_val_loss < best_loss:
                best_loss = average_val_loss
                best_model_wts = model.state_dict()
            print(f"Epoch: {epoch} --- Train loss: {average_train_loss:7.4f} --- Eval loss: {average_val_loss:7.4f}")
    print("Best loss:", best_loss)

    if best_model_wts is not None:
        torch.save(best_model_wts,"best_model.pth")
    # I want to save the model with the best loss

model = get_model()


train(model,32,torch.optim.Adam,10,0.001)

