import torch
import keras2onnx
from models.pytorch.mobilenet_pt import MobileNetv1 as MBN_pt
from models.pytorch.vgg_pt import VGG as VGG_pt
from models.tensorflow.mobilenet_tf import MobileNetv1 as mobilenet_tf
from models.tensorflow.vgg_tf import VGG11_tf as VGG_tf


def convert_pt():
    model = MBN_pt()
    model.load_state_dict(torch.load("mbnv1_pt.pt"))
    torch.onnx.export(model, torch.randn(1,3,32,32), f"mbnv1_pt.onnx", export_params=True, opset_version=10)

    model = VGG_pt()
    model.load_state_dict(torch.load("vgg_pt.pt"))
    torch.onnx.export(model, torch.randn(1,3,32,32), f"vgg_pt.onnx", export_params=True, opset_version=10)

def convert_tf():
    model = MBN_tf()
    model.lead_weights("mbnv1_tf.ckpt")
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras.onnx.save_model(onnx_model, f"mbnv1_tf.onnx")

    model = VGG_tf()
    model.lead_weights("vgg_tf.ckpt")
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras.onnx.save_model(onnx_model, f"vgg_tf.onnx")