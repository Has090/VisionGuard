import sys
import platform
import datetime
import tensorflow as tf
import PIL
import pandas as pd
import numpy as np
import matplotlib
import sklearn
import dvc
import mlflow

print(f"Executed by: {platform.node()} | Timestamp: {datetime.datetime.now()}")
print(f"Python Version     : {sys.version.split()[0]}")
print(f"TensorFlow Version : {tf.__version__}")
print(f"Pillow (PIL)       : {PIL.__version__}")
print(f"Pandas Version     : {pd.__version__}")
print(f"NumPy Version      : {np.__version__}")
print(f"Matplotlib Version : {matplotlib.__version__}")
print(f"Scikit-Learn       : {sklearn.__version__}")
print(f"DVC Version        : {dvc.__version__}")
print(f"MLflow Version     : {mlflow.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs     : {gpus}")
if gpus:
    print("GPU Status         : TensorFlow CAN see a GPU.")
else:
    print("GPU Status         : No dedicated CUDA GPU detected (Running on CPU / Integrated Graphics).")
