from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite

# TODO: add argument parser

# TODO: add one argument for selecting VGG or MobileNet-v1 models

# TODO: Modify the rest of the code to use the arguments correspondingly

tflite_model_name = "" # TODO: insert TensorFlow Lite model name

# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for filename in tqdm(os.listdir("HW3_files/test_deployment")):
  with Image.open(os.path.join("HW3_files/test_deployment", filename)).resize((32, 32)) as img:
    input_image = np.expand_dims(np.float32(img), axis=0)

    # Set the input tensor as the image
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run the actual inference
    interpreter.invoke()

    # Get the output tensor
    pred_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Find the prediction with the highest probability
    top_prediction = np.argmax(pred_tflite[0])

    # Get the label of the predicted class
    pred_class = label_names[top_prediction]