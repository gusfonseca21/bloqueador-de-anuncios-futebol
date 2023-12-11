import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import cv2

new_model = load_model(os.path.join('models', 'detector_modelo.h5'))

img = cv2.imread('teste_anuncio_globo.jpg')
yhat = new_model.predict(np.expand_dims(img/255, 0))

print(yhat)

if yhat > 0.5:
    print("Sem anuncio")
else:
    print('Anuncio')

