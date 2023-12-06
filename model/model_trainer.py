from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import imghdr
import cv2
import os

data = tf.keras.utils.image_dataset_from_directory('data', image_size=(32, 57))
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print(batch[1])

# 0 = ANUNCIO
# 1 = SEM ANUNCIO
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
# plt.show()

#CONTINUAR EM 30:00 https://www.youtube.com/watch?v=jztwpsIzEGc