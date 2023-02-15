import numpy as np
import matplotlib
import matplotlib.pyplot as plt

loss_train_history=np.load("train_loss.npy")
loss_test_history=np.load("val_loss.npy")
accuracy_train_history=np.load("train_acc.npy")
accuracy_test_history=np.load("val_acc.npy")
total_epochs=len(loss_train_history)

print(loss_train_history,loss_test_history,accuracy_train_history,accuracy_test_history)

plt.plot(range(1,total_epochs+1),loss_train_history,label="train loss")
plt.plot(range(1,total_epochs+1),loss_test_history,label="test loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs")
plt.legend()

plt.savefig("plots/" + 'loss.png', dpi=600, facecolor="white", edgecolor='none')

plt.show()

plt.plot(range(1,total_epochs+1), accuracy_train_history, label='Train')
plt.plot(range(1,total_epochs+1), accuracy_test_history, label='Test')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epochs")
plt.legend()

plt.savefig("plots/" + 'accuracy.png', dpi=600, facecolor="white", edgecolor='none')

plt.show()