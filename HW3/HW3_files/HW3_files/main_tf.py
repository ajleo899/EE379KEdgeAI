import tensorflow as tf
import argparse
from models.tensorflow import vgg_tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.utils import to_categorical

# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

random_seed = 1
tf.random.set_seed(random_seed)

# TODO: Insert your model here
model = vgg_tf.VGG11_tf()
model.summary()

# TODO: Load the training and testing datasets
(X_train, y_train), (X_test, y_test)= cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
nb_classes = 10
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

## find channel mean, std and do data normalization
train_mean = (0.4914, 0.4822, 0.4465)
train_std = (0.2023, 0.1994, 0.2010)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

print(Y_train.shape)


# TODO: Convert the datasets to contain only float values

# TODO: Normalize the datasets

# TODO: Encode the labels into one-hot format

# TODO: Configures the model for training using compile method
model.compile(optimizer="Adam", loss="categorical_crossentropy")
# TODO: Train the model using fit method
history = model.fit( X_train, Y_train, batch_size = batch_size, epochs = epochs)
print(history.history)
# TODO: Save the weights of the model in .ckpt format
