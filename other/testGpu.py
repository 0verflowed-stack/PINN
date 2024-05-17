import tensorflow as tf
import os

print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print(os.environ["CUDA_VISIBLE_DEVICES"])

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

print("Available devices:", tf.config.list_physical_devices())

if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())