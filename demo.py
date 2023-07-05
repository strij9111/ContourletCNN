import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from contourlet_cnn import ContourletCNN
from torch.utils.data import Dataset
import subprocess


# Custom dataset class
class ListImages(Dataset):
    def __init__(self, img_paths, transforms=None, target_transform=None):
        self.img_paths = img_paths
        self.img_labels = np.zeros(len(self.img_paths))
        self.transforms = transforms
        self.target_transform = target_transform
           
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]

        command = [
            'ffmpeg', 
            '-loglevel', 'quiet', '-y',
            '-i', img_path, 
            '-lavfi', 'showspectrumpic=s=800x800:mode=separate:legend=disabled', 
            'spectrogram.png'
        ]

        subprocess.run(command, check=True)

        image = Image.open('spectrogram.png').convert("RGB")
        # Load label
        label = self.img_labels[idx]
        # Transform image or label
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
# Device settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 

# Initialize an instance of C-CNN model
model = ContourletCNN(input_dim=(3, 400, 400), n_levs=[0, 3, 3, 3], num_classes=2, variant="SSF", spec_type="all").to(device)
print(model)

# Load a single image
test_loader = torch.utils.data.DataLoader(
    ListImages([
        "e:\\phoneme_a\\b2.wav", 
        "e:\\phoneme_a\\b3.wav", 
        "e:\\phoneme_a\\fb.wav", 
    ], transforms=[
        transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
        ])
    ]), 
    batch_size=2, shuffle=False)

with torch.no_grad():
    for data, target in test_loader:
        x = data.to(device)
        # Here, we call the private method, __pdfbdec() from C-CNN on the image
        y, _ = model._ContourletCNN__pdfbdec(x=x, method="resize")
        
import matplotlib.pyplot as plt


# Collect tensor with same dimension into a dict of list
output = {}
for i in range(len(y)):
    if isinstance(y[i], list):
        for j in range(len(y[i])):
            shape = tuple(y[i][j].shape)
            if shape in output.keys():
                output[shape].append(y[i][j])
            else:
                output[shape] = [y[i][j]]
    else:
        shape = tuple(y[i].shape)
        if shape in output.keys():
            output[shape].append(y[i])
        else:
            output[shape] = [y[i]]

# Concat the list of tensors into single tensor
for k in output.keys():
    output[k] = torch.cat(output[k], dim=1)

    
for k in output.keys():
    print(k)
    idx = 1
    # create figure
    fig = plt.figure(figsize=(9, 9))
    
    for i in range(output[k].shape[0]):
        for j in range(output[k].shape[1]):
            # print(output[k][i][j].shape)
            fig.add_subplot(4, k[1]//4, idx)
            plt.axis('off')
            plt.imshow(output[k][i][j])
            idx = idx + 1
    plt.tight_layout()
    plt.show()
#    break
    
    