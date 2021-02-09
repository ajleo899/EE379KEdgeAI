import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import thop
from thop import profile
import torchsummary
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import time

# Argument parser
parser = argparse.ArgumentParser(description='EE397K HW1 - SimpleCNN')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

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
# TODO: Insert here the normalized MNIST dataset
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)]), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dataset Loader (Input Pipeline)
'''train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)'''


# Define your model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1) # 32 --> 4 output feature maps
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1) # 64 --> 8 output feature maps
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(7 * 7 * 8, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


#learning_rates = [0.01, 0.001, 0.0001]
#optimizers = ['SGD', 'RMS', 'ADAM']
'''for opt in optimizers:
    for rate in learning_rates:'''
batch_sizes = [1,2,8,128,256,2048]
times = []
for batch in batch_sizes:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)


    model = SimpleCNN(num_classes)
    model = model.to(device)
    print('BATCH SIZE {batch}'.format(batch = batch))
    #print('LEARNING RATE: ' + str(rate))
    # Define your loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    ''' if opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum=0.9, weight_decay=5e-4)
    elif opt == 'RMS':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=rate, alpha=0.99, eps=1e-8)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-08)'''

    epoch_losses = np.zeros(num_epochs)
    epoch_losses_test = np.zeros(num_epochs)
    accuracy_train = np.zeros(num_epochs)
    accuracy_test = np.zeros(num_epochs)

    t = time.perf_counter()

    for epoch in range(num_epochs):
        # Training phase loop
        train_correct = 0
        train_total = 0
        train_loss = 0
        # Sets the model in training mode.
        model = model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Sets the gradients to zero
            optimizer.zero_grad()
            # The actual inference
            outputs = model(images)
            # Compute the loss between the predictions (outputs) and the ground-truth labels
            loss = criterion(outputs, labels).to(device)
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
            if (batch_idx + 1) % ((len(train_dataset) // batch)/4) == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                                    len(train_dataset) // batch,
                                                                                    train_loss / (batch_idx + 1),
                                                                                    100. * train_correct / train_total))

        epoch_losses[epoch] = train_loss   
        accuracy_train[epoch] = 100. * train_correct / train_total
    ExecTime = time.perf_counter() - t
    print("Time to Train: " + str(ExecTime))
    times.append(ExecTime)                                                                                
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
            images, labels = images.to(device), labels.to(device)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels).to(device)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    print('----------------------------------------------------------------')
    epoch_losses_test[epoch] = test_loss
    accuracy_test[epoch] = 100. * test_correct / test_total

#time vs batch_size plot
a = plt.figure(1)
plt.xscale('log',basex=2)
plt.title('Batch Size vs Time to Train')
plt.xlabel('Batch Size')
plt.ylabel('Time to Train')
plt.grid(True)
plt.plot(batch_sizes, times)
a.savefig('simpleCNN_batchsize_time')
# Loss plots
'''x_val = np.arange(1, num_epochs+1)
a = plt.figure(1)
plt.plot(epoch_losses)
plt.plot(epoch_losses_test)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss: lr = {rate}, Optimizer = {optimizer}'.format(rate = rate, optimizer = opt))
plt.grid(True)
plt.legend(["Training", "Testing"], loc ="upper right")
#a.savefig('simpleCNN_loss_plots_before_reducing.png')
a.savefig('simpleCNN_loss_plots_{optimizer}_{rate}.png'.format(rate = rate, optimizer = opt))
plt.cla()'''

# Accuracy plots
'''c = plt.figure(2)
plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.title('Training and Test Accuracy After Reducing Complexity')
plt.grid(True)
plt.legend(["Training", "Testing"], loc ="lower right")
#c.savefig('simpleCNN_acc_plots_before_reducing.png')
c.savefig('simpleCNN_acc_plots_after_reducing.png')'''


# GMACs and GFLOPs
'''macs, params = profile(model, inputs=(torch.randn(1, 1, 28, 28).to(device), ))
print("MACS: " + str(macs))
print("GFLOPS: " + str(macs/2))
'''
# number of params, model size in MB
summary(model, (1, 28, 28))

# Saving model
torch.save(model.state_dict(), 'CNN_saved_model')
