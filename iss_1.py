import torch
import numpy as np
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

timestamp = time.strftime("%Y%m%d-%H%M")

data= np.load("D:/ams/mc.npz")
x=data['x']
y=data['y']
#print("shape=",x.shape,"view=", x.view(),"x[0]=", x[0])

#x, _, y, _ = train_test_split( x, y, test_size=0.8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4)

batch_size= 16
x_train = torch.from_numpy(x_train).to(torch.float32)
y_train = torch.from_numpy(y_train).to(torch.float32)

Train_Data = TensorDataset(x_train,y_train)
train_loader=DataLoader(Train_Data, batch_size=batch_size, shuffle=True)

x_test = torch.from_numpy(x_test).to(torch.float32)
y_test = torch.from_numpy(y_test).to(torch.float32)

Test_Data = TensorDataset(x_test,y_test)
test_loader=DataLoader(Test_Data, batch_size=batch_size, shuffle=True)

from MyFlexibleCNN import *
from eval import *
#mlp = MLP(18*72,256,1)
model=CNN2(img_size_x=18,img_size_y=72,in_chan=1,out_chan=64)
#device = torch.device('cpu')
name_trained_model= "cnn_20230213hfdghdghd-1841_epoch-1.pt"
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
limit_patience=30
for epoch in range(total_epochs):
    best_loss=100
    training(train_loader, optimizer, model, loss_function)

    loss=validation(model, test_loader, loss_function)
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

