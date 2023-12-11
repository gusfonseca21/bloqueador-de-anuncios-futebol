from tabnanny import verbose
import pyautogui
import cv2 as cv
import numpy as np
import time
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

new_model = load_model(os.path.join('.', 'model', 'models', 'detector_modelo.h5'))

fps = 10
interval = 1 / fps

screenWidth, screenHeight = pyautogui.size()
targetWidth, targetHeight = 1920, 1080

xOffset = (screenWidth - targetWidth) // 2
yOffset = (screenHeight - targetHeight) // 2

try:
    while True:
        screenshot = pyautogui.screenshot(region=(xOffset, yOffset, targetWidth, targetHeight))
        screenshotArray = np.array(screenshot)
        screenshotArray = cv.cvtColor(screenshotArray, cv.COLOR_RGB2BGR)

        screenshotShrinked = cv.resize(screenshotArray, (57, 32), interpolation=cv.INTER_AREA)

        yhat = new_model.predict(np.expand_dims(screenshotShrinked/255, 0), verbose = 0)

        if yhat > 0.5:
            print("Sem anuncio")
        else:
            print('Anuncio')

        time.sleep(interval)
except KeyboardInterrupt:
    print("Parou de analisar imagens")