import torch
import numpy as np
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import os.path as path
from torch.utils.data import DataLoader, TensorDataset
import time
from tempfile import mkdtemp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
import random


timestamp = time.strftime("%Y%m%d-%H%M")
""""
data= np.load("D:/ams/mc.npz")
x=data['x']
y=data['y']
"""
data_path = "D:/ams/mc.npz"

with np.load(data_path, mmap_mode="r") as data:
    if isinstance(data["x"], np.memmap):
        x_memmap = np.memmap(data["x"].base, mode="r", shape=data["x"].shape)
    else:
        x_memmap = data["x"]
    if isinstance(data["y"], np.memmap):
        y_memmap = np.memmap(data["y"].base, mode="r", shape=data["y"].shape)
    else:
        y_memmap = data["y"]

indices = random.sample(range(len(x_memmap)), 1000)
x = x_memmap[indices]
y = y_memmap[indices]

#print("shape=",x.shape,"view=", x.view(),"x[0]=", x[0])

#x, _, y, _ = train_test_split( x, y, test_size=0.8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4)

batch_size= 128
x_train = torch.from_numpy(x_train).to(torch.float32)
y_train = torch.from_numpy(y_train).to(torch.float32)

Train_Data = TensorDataset(x_train,y_train)
train_loader=DataLoader(Train_Data, batch_size=batch_size, shuffle=True)

x_test = torch.from_numpy(x_test).to(torch.float32)
y_test = torch.from_numpy(y_test).to(torch.float32)

Test_Data = TensorDataset(x_test,y_test)
test_loader=DataLoader(Test_Data, batch_size=batch_size, shuffle=True)

from MyFlexibleCNN import CNN2
from eval import *
#mlp = MLP(18*72,256,1)
model=CNN2(img_size_x=18,img_size_y=72,in_chan=1,out_chan=64)
#device = torch.device('cpu')
name_trained_model= "cnn2_20230214-1453kgkgkfg_epoch-3.pt"
if (os.path.isfile(name_trained_model)):
    checkpoint = torch.load(name_trained_model) if torch.cuda.is_available() else torch.load(name_trained_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict']) if os.path.isfile(name_trained_model) else print(" ")
    print("trained model is uploaded")

#if torch.cuda.device_count() > 1:
  #  print("Number of GPUs being used:", torch.cuda.device_count())
  #  model = nn.DataParallel(model)  #2

#model.to(device)


loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
model_name= "cnn2"

total_epochs=300
patience=0
limit_patience=10
best_loss = 100

for epoch in range(total_epochs):

    training(train_loader, optimizer, model, loss_function)

    loss = validation(model, test_loader, loss_function)
    print(loss)
    if (loss < best_loss):
        patience = 0
        best_loss = loss
        print("best loss changed", best_loss, "patience ", patience)
        try:
            state_dict = model.module.state_dict()  # To unwrap DataParallel model.
        except AttributeError:
            state_dict = model.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, "checkpoints/" + model_name + "_" + timestamp +
                   "_epoch-{}.pt".format(epoch + 1))
        ######
        ### just to use in the plotting
        a = [(epoch + 1), best_loss]
        np.save("early_stopping.npy", a)
    else:
        patience = patience + 1
        if (patience == limit_patience):
            break
        elif (patience < limit_patience):
            print("patience increased to", patience)

