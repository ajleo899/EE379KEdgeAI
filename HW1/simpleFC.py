import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

# Argument parser
parser = argparse.ArgumentParser(description='EE397K HW1 - SimpleFC')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

epoch_losses = np.zeros(num_epochs)
epoch_losses_test = np.zeros(num_epochs)
accuracy_train = np.zeros(num_epochs)
accuracy_test = np.zeros(num_epochs)

for epoch in range(num_epochs):
    # Training phase loop
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    start = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    
    epoch_losses[epoch] = train_loss   
    accuracy_train[epoch] = 100. * train_correct / train_total
    end = time.time()
    print("Time to Train: " + str(end - start))
    #print(epoch_losses)

    # Testing phase loop
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    epoch_losses_test[epoch] = test_loss
    accuracy_test[epoch] = 100. * test_correct / test_total

x_val = np.arange(1, num_epochs+1)
a = plt.figure(1)
plt.plot(epoch_losses)
plt.plot(epoch_losses_test)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True)
plt.legend(["Training", "Testing"], loc ="upper right")
a.savefig('training_testingloss_FC.png')

c = plt.figure(2)
plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.title('Training and Test Accuracy')
plt.grid(True)
plt.legend(["Training", "Testing"], loc ="lower right")
c.savefig('training_testingaccuracy_FC.png')
input()

#plt.close('all')