import pyautogui
import cv2 as cv
import numpy as np
import time
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

fps = 3
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


        cv.imwrite(f'./images/lake/TESTE_{time.time()}.jpg', screenshotShrinked)

        time.sleep(interval)
        print(f"Captura feita_{time.time()}")
except KeyboardInterrupt:
    print("Parou de capturar screenshots")