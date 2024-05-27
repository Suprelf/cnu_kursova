import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_image(path):

    img = plt.imread(path)  # cv2.IMREAD_GRAYSCALE)
    size = img.shape
    dimension = (size[0], size[1])

    return img[...,::-1], size, dimension

print("Select first image:")
Tk().withdraw()
filename1 = askopenfilename()
img1, size1, dimension1 = read_image(filename1)
img1 = Image.fromarray(img1.astype('uint8'))
print(f"Image selected: {os.path.basename(os.path.normpath(filename1))}")

print("Select second image:")
Tk().withdraw()
filename2 = askopenfilename()
img2, size2, dimension2 = read_image(filename2)
img2 = Image.fromarray(img2.astype('uint8'))
print(f"Image selected: {os.path.basename(os.path.normpath(filename2))}")

print(f"Calculated error: {
    np.average(abs(
        np.array(img1, dtype="int16") -
        np.array(img2, dtype="int16"))
    )}")