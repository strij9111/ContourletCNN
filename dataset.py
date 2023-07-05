from PIL import Image
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset
import random
import hashlib
import subprocess
from torchvision import datasets, transforms
import librosa
from scipy.io.wavfile import write
from contourlet_cnn import ContourletCNN


# Custom dataset class
class ListImages(Dataset):
    def __init__(self, img_paths, transforms=None, target_transform=None):
        self.img_paths = img_paths
        self.img_labels = np.zeros(len(self.img_paths))
        self.transforms = transforms
        self.target_transform = target_transform
        self.cache_spectrogram = {}
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx].split('|')
        random_number = hashlib.md5(img_path[0].encode()).hexdigest()
        filename = f'spectrogram_{random_number}.png'
        
        if not os.path.isfile(filename):

            y, sr = librosa.load(img_path[0], sr=None)   
            frame_length = int(sr * 0.2)  # длина окна в отсчетах (200 мс)
            hop_length = 100  # шаг между окнами

            # Вычисление энергии сигнала в каждом окне
            energy = np.array([
                np.sum(np.abs(y[i:i+frame_length]**2))
                for i in range(0, len(y), hop_length)
            ])            

            max_energy_frame = np.argmax(energy)

            # Получение отсчетов этого окна
            start = max_energy_frame * hop_length
            end = start + frame_length

            max_energy_signal = y[start:end]

            # Преобразование сигнала обратно в 16-битные целые числа
            max_energy_signal_int16 = np.int16(max_energy_signal / np.max(np.abs(max_energy_signal)) * 32767)

            # Сохранение сигнала в новый WAV-файл
            write(f"{random_number}.wav", sr, max_energy_signal_int16)
            
            command = [
                'ffmpeg', 
                '-loglevel', 'quiet', '-y',
                '-i', f"{random_number}.wav", 
                '-lavfi', 'showspectrumpic=s=800x800:mode=separate:legend=disabled', 
                filename
            ]

            subprocess.run(command, check=True)
            os.remove(f"{random_number}.wav")
            
        image = Image.open(filename).convert("RGB")
        
        # Transform image or label
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        if self.target_transform:
            label = self.target_transform(label)
                
#            os.remove(filename)
        
        label = int(img_path[1])-35

        return image, label