import torch
import os.path
import numpy as np
def training(train_loader, optimizer, cnn2, loss_function):

    loss_sum = 0
    total_event = 0
    correct = 0
    total = 0
    # for batch in train_loader:
    cnn2.train()
    for i, (images, labels) in enumerate(train_loader):

        # images,labels=batch
        optimizer.zero_grad()
        predictions = cnn2(images)
        loss = loss_function(predictions, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        total_event += len(labels)

        predictions = torch.sigmoid(predictions)
        correct += (predictions.round() == labels).sum().item()
        total += len(labels)

    avg_loss = (loss_sum / total)

    #accuracy_train_history.append((100 * correct) / (total))
    #loss_train_history.append((loss_sum / total_event))
    acc = (100 * correct) / (total)
    if os.path.isfile("train_loss.npy"):
        y = np.load("train_loss.npy")
    else:
        y = []
    np.save("train_loss.npy", np.append(y, avg_loss))

    if os.path.isfile("train_acc.npy"):
        y = np.load("train_acc.npy")
    else:
        y = []
    np.save("train_acc.npy", np.append(y, acc))

        # make zero if doess not work
def validation(cnn2, test_loader,loss_function):
    cnn2.eval()
    loss_sum = 0
    total_event = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):

        out = cnn2(images)
        predictions = torch.sigmoid(out)

        loss = loss_function(predictions, labels)

        loss_sum += loss.item()
        total_event += len(labels)

        correct += (predictions.round() == labels).sum().item()
        total += len(labels)

    acc=(100 * correct) / (total)
    avg_loss = (loss_sum / total)
    #accuracy_test_history.append((100 * correct) / (total))
    # print('Epoch [%d/%d], test Accuracy: %.3f %%' % ((100 * correct) / (total)))
    #loss_test_history.append((loss_sum / total_event))
    if os.path.isfile("val_loss.npy"):
        y = np.load("val_loss.npy")
    else:
        y = []
    np.save("val_loss.npy", np.append(y, avg_loss))

    if os.path.isfile("val_acc.npy"):
        y = np.load("val_acc.npy")
    else:
        y = []
    np.save("val_acc.npy", np.append(y, acc))
    return avg_loss


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
