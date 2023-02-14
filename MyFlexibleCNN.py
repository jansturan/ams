import torch.nn as nn
import torch.nn.functional as func
import torch
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # <<< YOUR CODE HERE >>> #
    # noinspection PyTypeChecker
    self.conv1 = nn.Conv2d(1, 16, 3)
    # noinspection PyTypeChecker
    self.conv2 = nn.Conv2d(16, 16, 3)
    self.relu= nn.ReLU()
    self.linear = nn.Linear(16*14*68,1)
    # <<< YOUR CODE HERE >>> #

  def forward(self, input):
    # <<< YOUR CODE HERE >>> #
    output = input ## 1x18x72

    output = self.relu(self.conv1(output)) #16x16x70
    output = self.relu(self.conv2(output)) #32x14x68
    output = output.view(-1, 16* 14 * 68)
    output=self.linear(output)


    return torch.flatten(output)

#cnn =Net()
#z=torch.zeros(1,1,18,72)
#print(cnn(z))

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(1, 64, 3)

    self.conv2 = nn.Conv2d(64, 128, 3)
    self.relu= nn.ReLU()
    self.linear = nn.Linear(128*14*68,1)
    # <<< YOUR CODE HERE >>> #

  def forward(self, input):
    # <<< YOUR CODE HERE >>> #
    output = input ## 1x18x72

    output = self.relu(self.conv1(output)) #16x16x70
    print(output.shape)
    output = self.relu(self.conv2(output)) #32x14x68
    print(output.shape)
    output = output.view(-1, 128* 14 * 68)
    print(output.shape)
    output=self.linear(output)
    print(output.shape)


    return torch.flatten(output)

x=torch.zeros(128,1,18,72)

cnn = CNN()
out =cnn(x)
print(out.shape)


class CNN2(nn.Module):
  def __init__(self,img_size_x,img_size_y,in_chan,out_chan):
    super().__init__()

    padding=0
    stride = 1
    kernel_size=3
    self.out_chan=out_chan
    self.conv1 = nn.Conv2d(in_chan, 4*out_chan, kernel_size=kernel_size,padding=padding,stride=stride)
    img_size_x = int((img_size_x + 2*padding-kernel_size)/stride+1)
    img_size_y = int((img_size_y + 2*padding-kernel_size)/stride+1)




    padding=0
    stride = 1
    kernel_size=3
    self.conv2 = nn.Conv2d(4*out_chan, out_chan, kernel_size=kernel_size,padding=padding,stride=stride)
    self.img_size_x = int((img_size_x + 2*padding-kernel_size)/stride+1)
    self.img_size_y = int((img_size_y + 2*padding-kernel_size)/stride+1)

    self.relu= nn.ReLU()
    self.linear = nn.Linear(out_chan*self.img_size_x*self.img_size_y,1)
    # <<< YOUR CODE HERE >>> #

  def forward(self, input):
    # <<< YOUR CODE HERE >>> #
    output = input ## 1x18x72
    output = self.relu(self.conv1(output)) #16x16x70
    output = self.relu(self.conv2(output)) #32x14x68
    #b,o,h,w=output.shape
    output = output.view(-1, self.out_chan*self.img_size_x*self.img_size_y)
    output=self.linear(output)


    return torch.flatten(output)

print("nhgjygj")
cnn2 = CNN2(img_size_x=18,img_size_y=72,in_chan=1,out_chan=64)

out =cnn2(x)
print(out.shape)