import numpy as np
import scipy as sp
import os 
import cv2 
import shutil
import random
import torch #pytorch 
#import matplotlib.pyplot as plt 
from torchvision import datasets, transforms
import torch.nn as nn #Neural Network
import torch.nn.functional as F #for nn functions(Argmax, Relu, Sigmoid, cross entropy etc...)
import torch.optim as optim # for optimizer Adam
import wandb #for weights and biases to see model train in real time
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
#from torchsummary import summary # this torch print summary of model
import sys
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
sys.path.append(os.getcwd())
from digi_face_dataset import DigiFaceDataset
import timm
class MyModel(nn.Module):
      
        #Data and Organization
        path = os.listdir("/Users/iandeleon/Documents/GitHub/INITtoWinIt/")
        print(os.listdir())
        ['.DS_Store', 'FaceID.ipynb', 'Split Folder', 'Faces Data']
        data_path = "/Users/iandeleon/Documents/GitHub/INITtoWinIt/"

        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        test_path = os.path.join(data_path, "test")

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            train_folder_path = os.path.join(train_path, folder_name)
            val_folder_path = os.path.join(val_path, folder_name)
            test_folder_path = os.path.join(test_path, folder_name)

            if not os.path.exists(train_folder_path):
                os.makedirs(train_folder_path)
            if not os.path.exists(val_folder_path):
                os.makedirs(val_folder_path)
            if not os.path.exists(test_folder_path):
                os.makedirs(test_folder_path)

            images = os.listdir(folder_path)
            random.shuffle(images)

            num_images = len(images)
            train_end = int(num_images * 0.7)
            val_end = int(num_images * 0.85)

            train_images = images[:train_end]
            val_images = images[train_end:val_end]
            test_images = images[val_end:]

            for image_name in train_images:
                src_path = os.path.join(folder_path, image_name)
                dst_path = os.path.join(train_folder_path, image_name)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)

            for image_name in val_images:
                src_path = os.path.join(folder_path, image_name)
                dst_path = os.path.join(val_folder_path, image_name)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)

            for image_name in test_images:
                src_path = os.path.join(folder_path, image_name)
                dst_path = os.path.join(test_folder_path, image_name)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)     
        """def train(model, dataloader, criterion, optimizer, device):
            model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            return running_loss / len(dataloader.dataset)
        def validate(model, dataloader, criterion, device):
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
            return running_loss / len(dataloader.dataset)"""
        def __init__(self, num_classes=2):
            super(MyModel, self).__init__()
            self.base_model = timm.create_model("efficientnet_b0", pretrained=False)
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(in_features, num_classes)
        def forward(self, x):
            return self.base_model(x)
        def test_model(model, loaded_weights, device): #test_data_path, transform
            model.load_state_dict(loaded_weights, strict=False)
            model.to(device)
            model.eval()

            correct_predictions = 0
            total_predictions = 0

        data_transforms = {
            'train': Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        train_path = "/Users/iandeleon/Documents/GitHub/INITtoWinIt/dataFiles"
        val_path = "/Users/iandeleon/Documents/GitHub/INITtoWinIt/dataFiles"

        train_dataset = DigiFaceDataset(train_path, transform=data_transforms['train'])
        val_dataset = DigiFaceDataset(val_path, transform=data_transforms['val'])

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(train_dataset.classes))
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 10
        #Loaded pretrained weights for efficientnet-b0
        """best_val_loss = float('inf')
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = train(model, train_dataloader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss = validate(model, val_dataloader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_efficientnet_b0.pth')
                print("Model saved.")"""
if __name__ == "__main__":
    model = MyModel()
    model.train()
