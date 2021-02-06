import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

# Argument parser
parser = argparse.ArgumentParser(description='EE397K HW1 - Starter code')
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
print(len(train_loader))


#train_loader = train_loader.to(device)
#test_loader = test_loader.to(device)

# Define your model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes)
model = model.to(device)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

epoch_losses = np.zeros(num_epochs)
epoch_losses_test = np.zeros(num_epochs)
accuracy_train = np.zeros(num_epochs)
accuracy_test = np.zeros(num_epochs)

'''starter1, ender1 = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
timings1 = 0

starter1.record()'''

t = time.perf_counter()
print('start timer')
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
        #print(batch_idx)
        images = images.view(-1, input_size)
        #print(type(images))
        images = images.to(device)
        labels = labels.to(device)
        #print('Device switched to GPU')
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels).to(device)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        #if (batch_idx + 1) % 100 == 0:
            #print('Back prop done')
        #print('Back Prop Complete')
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        #print('training completed')
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        #print('about to print')
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    #print(epoch)
    epoch_losses[epoch] = train_loss   
    accuracy_train[epoch] = 100. * train_correct / train_total
    end = time.time()
    #print("Time to Train: " + str(end - start))
    #print(epoch_losses)
ExecTime = time.perf_counter() - t
print("Time to Train: " + str(ExecTime))
print('Train timer complete')
'''ender1.record()
torch.cuda.synchronize()
curr_time = starter1.elapsed_time(ender1)
timings1 = curr_time
print('The total time to train: ' + str(timings1))'''

# Testing phase loop
test_correct = 0
test_total = 0
test_loss = 0
# Sets the model in evaluation mode
model = model.eval()
# Disabling gradient calculation is useful for inference.
# It will reduce memory consumption for computations.

#Timings for when GPU is in use
#starter2 = torch.cuda.Event(enable_timing=True)
#ender2 = torch.cuda.Event(enable_timing=True)
#Timings for CPU is in use
#timings2=np.zeros((len(test_loader),1))
#timings_cpu = np.zeros((len(test_loader),1))

t2 = time.perf_counter()

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        #FOR GPU
        #starter2.record()
        #FOR CPU
        #starter_cpu = time.time()
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        images = images.to(device)
        labels = labels.to(device)
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
        #FOR CPU
        #ender_cpu = time.time()
        #timings_cpu[batch_idx] = ender_cpu - starter_cpu
        #FOR GPU
        #ender2.record()
        #torch.cuda.synchronize()
        #curr_time = starter.elapsed_time(ender2)
        #timings2[batch_idx] = curr_time


#Get Average for GPU Timings
#mean_syn = np.sum(timings2) / len(test_loader)
#std_syn = np.std(timings2)

#print("Average Inference Time per Image -- GPU: " + str(mean_syn)) 
#print("Average Inference Time per Image -- CPU: " + str(np.sum(timings_cpu) / len(test_loader))) 
ExecTime2 = time.perf_counter() - t2
print('Time for all inferences: ' + str(ExecTime2))
print('Time for single image inference (average): ' + str(ExecTime2 / len(test_loader)))
print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
epoch_losses_test[epoch] = test_loss
accuracy_test[epoch] = 100. * test_correct / test_total

'''x_val = np.arange(1, num_epochs+1)
a = plt.figure(1)
plt.plot(epoch_losses)
#a.show()
#a.savefig('trainingloss.png')
plt.plot(epoch_losses_test)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True)
plt.legend(["Training", "Testing"], loc ="upper right")
a.savefig('training_testingloss_LR.png')
#b.show()
#b.savefig('testingloss.png')
c = plt.figure(2)
plt.plot(accuracy_train)
#a.show()
#a.savefig('trainingloss.png')
plt.plot(accuracy_test)
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.title('Training and Test Accuracy')
plt.grid(True)
plt.legend(["Training", "Testing"], loc ="lower right")
c.savefig('training_testingaccuracy_LR.png')
input()

#plt.close('all')
'''