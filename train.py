import os
import time
import numpy as np
from collections import defaultdict
import random
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from dataset import ListImages
from contourlet_cnn import ContourletCNN
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


torch.cuda.empty_cache()

def train(model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
#        pbar.update(batch_idx, values=[("loss", loss.item())])
    return train_loss

            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target.long(), reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct
    
def setup():
    ###### Training settings
    n_epochs = 70
    val_size = 0.2
    lr = 0.001
    seed = 2021
    img_dim = (224, 224)
    batch_size = 4

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    #%%

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # DataLoaders
    if use_cuda:
        num_workers = 2
        pin_memory = True
    else:
        num_workers = 2
        pin_memory = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    base_dir = "e:\\phoneme_a\\wavs"
    file_path = "dataset.txt"

    # Словарь для группировки данных по ID спикеров
    data_dict = defaultdict(list)

    # Чтение данных и группировка по ID спикеров
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            filename, speaker_id = line.strip().split('|')
            data_dict[speaker_id].append(base_dir + "\\" + line)

    # Списки для обучающей и тестовой выборок
    train_list = []
    test_list = []

    # Выбор 1-2 записей для каждого спикера для тестовой выборки
    for speaker_id, filenames in data_dict.items():
        # Перемешиваем список файлов
        random.shuffle(filenames)
        # Если у спикера более двух записей, выбираем две для теста, остальные - для обучения
        if len(filenames) > 2:
            test_list.extend(filenames[:2])
            train_list.extend(filenames[2:])
        # Если у спикера две записи, выбираем одну для теста, одну - для обучения
        elif len(filenames) == 2:
            test_list.append(filenames[0])
            train_list.append(filenames[1])
        # Если у спикера одна запись, добавляем ее в обе выборки
        else:
            test_list.append(filenames[0])
            train_list.append(filenames[0])


    train_loader = torch.utils.data.DataLoader(
        ListImages(train_list, transforms=[
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        ]), 
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)
        
    test_loader = torch.utils.data.DataLoader(
        ListImages(test_list, transforms=[
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        ]), 
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                               pin_memory=pin_memory)
                                               
    model = ContourletCNN(input_dim=(3, 224, 224), num_classes=30, variant="SSF", spec_type="all").to(device)
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters", params)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=0, amsgrad=False)

    start_time = time.perf_counter()
    for epoch in range(n_epochs):
        n_batches = len(train_loader)
    #    pbar = Progbar(target=n_batches)
        print(f'Epoch {epoch+1}/{n_epochs}')
        train_loss = train(model, device, train_loader, optimizer, epoch+1)
        test_loss, correct = test(model, device, test_loader)

        print(f"Loss: {test_loss}, accuracy: {100. * correct / len(test_loader.dataset):.0f}%")
    end_time = time.perf_counter()
    print(end_time - start_time, "seconds")

if __name__ == '__main__':
    setup()