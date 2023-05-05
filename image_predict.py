import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img_path = "/Users/srisruthi/Downloads/20220607T154733.JPG"
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
img_array = image.img_to_array(img)
print(img_array)
img_batch = np.expand_dims(img_array, axis=0)
