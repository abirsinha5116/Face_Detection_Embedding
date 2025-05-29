import os
import tensorflow as tf

cuda_home = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
os.environ["CUDA_PATH"] = cuda_home
os.environ["PATH"] = (
    os.path.join(cuda_home, "bin") + ";" +
    os.path.join(cuda_home, "libnvvp") + ";" +
    os.environ["PATH"]
)
# Called at top of every script
def enable_memory_growth():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
