# Author: Jack Haek
# Dataset Used: MNIST
# CPU implementation of training a Feed-Forward Neural Network
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

# Specific for the PyTorch MNIST Dataset
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

DATA_ROOT = "data"
BATCH_SIZE = 16
LEARN_RATE = 0.001
EPOCHS = 3

training_data = datasets.MNIST(root=DATA_ROOT, train = True, download = True, transform=ToTensor())
testing_data = datasets.MNIST(root=DATA_ROOT, train = False, download = True, transform=ToTensor())

# Import the data so we can actually use it
train_set = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle = True)
test_set = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle = True)

# Look at any image in the first batch.
# Note that IMAGE_NUMBER is zero indexed
IMAGE_NUMBER = 0
for data in train_set:
    print(data)
    break

if 0 <= IMAGE_NUMBER < BATCH_SIZE and isinstance(IMAGE_NUMBER, int):
    x, y = data[0][IMAGE_NUMBER], data[1][IMAGE_NUMBER]
    plt.imshow(x.view(28, 28), cmap = "Greys")
    print("The image is a " + str(int(y)))

# Check the balance of the data
total = 0
counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
percentages = []
x_axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for data in tqdm(train_set):
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

for i in counter_dict:
    percentages.append(counter_dict[i] / total * 100)

plt.bar(x_axis, percentages)
plt.title("Distribution of Data")
plt.xlabel("Classification")
plt.ylabel("Percentage of Data")
plt.show()


# Build the network
class Net(nn.Module):
    def __init__(self):
        # Initialize the nn.Module class
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # Images are 28 by 28 so that is the input for the first layer
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # 10 is the output because we have 10 separate classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
# Define the optimizer algorithm. PyTorch comes with a few. They are generally good
# Note that the time.sleep functions are used so that tqdm displays correctly
optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
train_losses = []
train_accuracies = []

test_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    for data in tqdm(train_set):
        # data is the whole batch
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    time.sleep(0.5)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(train_set):
            X, y = data
            output = net(X.view(-1, 28*28))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    train_losses.append(float(loss))
    train_accuracies.append(float(correct/total))
    time.sleep(0.5)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_set):
            X, y = data
            output = net(X.view(-1, 28*28))
            loss = F.nll_loss(output, y)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    test_losses.append(float(loss))
    accuracy = correct/total
    test_accuracies.append(accuracy)
    print(f"Train Loss: {round(train_losses[-1], 3)}")
    print(f"Test Loss: {round(test_losses[-1], 3)}")
    print(f"Train Accuracy: {round(train_accuracies[-1]*100, 3)}")
    print(f"Test Accuracy: {round(test_accuracies[-1]*100, 3)}")
    time.sleep(0.5)

x_axis = np.linspace(1, EPOCHS, EPOCHS, dtype=int)
plt.plot(x_axis, test_losses, color='Red', label = 'Test Loss')
plt.plot(x_axis, train_losses, color='Black', label = 'Train Loss')
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(x_axis, test_accuracies, color='Red', label = 'Test Loss')
plt.plot(x_axis, train_accuracies, color='Black', label = 'Train Loss')
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
