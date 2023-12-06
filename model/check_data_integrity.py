from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import imghdr
import cv2
import os

data_dir = 'data'
image_exts = ["jpeg"]

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Imagem com extensão errada {}".format(image_exts))
                os.remove(image_path)
        except Exception as e:
            print("Há algum problema com a image {}".format(image_path))
            # os.remove(image_path)