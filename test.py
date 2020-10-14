import tensorflow as tf
from tensorflow import keras
from lib.network import Network_restored
import lib.dataset as dataset

path = "/home/mitesh/3D-R2N2-tf_v2/models_local/model_2020-08-31_23:34:47/epoch_99"



model = Network_restored(path)

# get preprocessed data
#data, label = dataset.load_preprocessed_dataset()
data =

out = model.predict(data)