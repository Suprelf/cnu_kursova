import os
import time
import tkinter as tk
from math import sqrt, floor
from tkinter.filedialog import askopenfilename

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_image(path):
    img = plt.imread(path)  # cv2.IMREAD_GRAYSCALE)
    size = img.shape
    dimension = (size[0], size[1])

    return img[..., ::-1], size, dimension


def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    scale /= 100
    new_dimension = (int(dimension[1] * scale), int(dimension[0] * scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)

    return resized_img


def nearest_interpolation(image, dimension):
    start_time = time.time()
    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    enlarge_time = int(
        sqrt((dimension[0] * dimension[1]) / (image.shape[0] * image.shape[1])))

    for i in range(dimension[0]):
        for j in range(dimension[1]):
            row = floor(i / enlarge_time)
            column = floor(j / enlarge_time)

            new_image[i, j] = image[row - 1, column - 1]

    end_time = time.time()
    print(f"Nearest time: {round(end_time - start_time, 4)}")

    return new_image[..., ::-1]


def bilinear_interpolation(image, dimension):
    start_time = time.time()

    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width) / (dimension[1])
    scale_y = (height) / (dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j + 0.5) * (scale_x) - 0.5
                y = (i + 0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width - 2)
                y_int = min(y_int, height - 2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int + 1, k]
                c = image[y_int + 1, x_int, k]
                d = image[y_int + 1, x_int + 1, k]

                pixel = a * (1 - x_diff) * (1 - y_diff) + b * (x_diff) * \
                        (1 - y_diff) + c * (1 - x_diff) * (y_diff) + d * x_diff * y_diff

                new_image[i, j, k] = pixel.astype(np.uint8)

    end_time = time.time()
    print(f"Bilinear time: {round(end_time - start_time, 4)}")

    return new_image[..., ::-1]


def W(x):
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a + 2) * (pos_x ** 3)) - ((a + 3) * (pos_x ** 2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x ** 3)) - (5 * a * (pos_x ** 2)) + (8 * a * pos_x) - 4 * a)
    else:
        return 0


def bicubic_interpolation(img, dimension):
    start_time = time.time()

    nrows = dimension[0]
    ncols = dimension[1]

    output = np.zeros((nrows, ncols, img.shape[2]), np.uint8)
    for c in range(img.shape[2]):
        for i in range(nrows):
            for j in range(ncols):
                xm = (i + 0.5) * (img.shape[0] / dimension[0]) - 0.5
                ym = (j + 0.5) * (img.shape[1] / dimension[1]) - 0.5

                xi = floor(xm)
                yi = floor(ym)

                u = xm - xi
                v = ym - yi

                out = 0
                for n in range(-1, 3):
                    for m in range(-1, 3):
                        if ((xi + n < 0) or (xi + n >= img.shape[0]) or (yi + m < 0) or (yi + m >= img.shape[1])):
                            continue

                        out += (img[xi + n, yi + m, c] * (W(u - n) * W(v - m)))

                output[i, j, c] = np.clip(out, 0, 255)

    end_time = time.time()
    print(f"Bicubic time: {round(end_time - start_time, 5)}")

    return output[..., ::-1]


def main():
    mode = input("Select mode: \nUpscale\nResearch\nCompare\n-> ").lower()

    if (mode == "upscale"):
        print("Upscale mode")

        # Select File
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        filename = askopenfilename()
        print(filename)

        # Read Image
        img, size, dimension = read_image(filename)  # "./test.jpg"
        print(f"Image size is: {size}")

        # Change Image Size
        scale_percent = int(input("Enter upscale size in %: "))  # percent of original image size
        resized_img = image_change_scale(img, dimension, scale_percent)
        print(f"Bigger Image size is: {resized_img.shape}")
        print(f"Worked pixels: {resized_img.shape[0] * resized_img.shape[1]}")
        dimension = np.array(dimension)
        dimension = dimension * (int(scale_percent / 100))

        fig, axs = plt.subplots(1, 1)

        nn_img_algo = nearest_interpolation(img, dimension)
        nn_img_algo = Image.fromarray(nn_img_algo.astype('uint8'))
        nn_img_algo.save(str(scale_percent) + "_n_" + os.path.basename(os.path.normpath(filename)))

        bil_img_algo = bilinear_interpolation(img, dimension)
        bil_img_algo = Image.fromarray(bil_img_algo.astype('uint8'))
        bil_img_algo.save(str(scale_percent) + "_l_" + os.path.basename(os.path.normpath(filename)))

        cubic_img_algo = bicubic_interpolation(img, dimension)
        cubic_img_algo = Image.fromarray(cubic_img_algo.astype('uint8'))
        cubic_img_algo.save(str(scale_percent) + "_b_" + os.path.basename(os.path.normpath(filename)))

        axs[0, 0].set_title("Original")
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        axs[0, 1].set_title("Nearest")
        axs[0, 1].imshow(nn_img_algo)

        axs[0, 0].set_title("Bilinear")
        axs[1, 1].imshow(bil_img_algo)

        axs[1, 0].set_title("Bicubic")
        axs[1, 0].imshow(cubic_img_algo)

        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if (mode == "research"):
        print("Research mode")

        # Select File
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        filename = askopenfilename()
        print(filename)

        # Read Image
        img, size, dimension = read_image(filename)  # "./test.jpg"
        print(f"Image size is: {size}")

        # Change Image Size
        scale_percent = 25  # percent of original image size
        resized_img = image_change_scale(img, dimension, scale_percent)
        print(f"Smalled Image size is: {resized_img.shape}")
        print(f"Worked pixels: {resized_img.shape[0] * resized_img.shape[1]}")

        fig, axs = plt.subplots(2, 3)

        nn_img_algo = nearest_interpolation(resized_img, dimension)
        nn_img_algo = Image.fromarray(nn_img_algo.astype('uint8'))
        nn_img_algo.save("n_" + os.path.basename(os.path.normpath(filename)))

        bil_img_algo = bilinear_interpolation(resized_img, dimension)
        bil_img_algo = Image.fromarray(bil_img_algo.astype('uint8'))
        bil_img_algo.save("l_" + os.path.basename(os.path.normpath(filename)))

        cubic_img_algo = bicubic_interpolation(resized_img, dimension)
        cubic_img_algo = Image.fromarray(cubic_img_algo.astype('uint8'))
        cubic_img_algo.save("b_" + os.path.basename(os.path.normpath(filename)))

        axs[0, 0].set_title("Original")
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        axs[1, 0].set_title(f"Smaller: {scale_percent}%")
        axs[1, 0].imshow(cv2.cvtColor(np.array(resized_img), cv2.COLOR_BGR2RGB))

        axs[0, 1].set_title("Nearest")
        axs[0, 1].imshow(nn_img_algo)

        axs[0, 2].set_title("Bilinear")
        axs[0, 2].imshow(bil_img_algo)

        axs[1, 1].set_title("Bicubic")
        axs[1, 1].imshow(cubic_img_algo)

        axs[1, 2].set_title("Lanczos (TODO)")
        axs[1, 2].imshow(cv2.cvtColor(np.array(resized_img), cv2.COLOR_BGR2RGB))

        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if (mode == "compare"):
        print("Compare mode")

        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        filename1 = askopenfilename()
        print(filename1)

        img1, size1, dimension1 = read_image(filename1)
        img1 = Image.fromarray(img1.astype('uint8'))
        print(f"Image selected: {os.path.basename(os.path.normpath(filename1))}")

        print("Select second image:")
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        filename2 = askopenfilename()
        print(filename2)

        img2, size2, dimension2 = read_image(filename2)
        img2 = Image.fromarray(img2.astype('uint8'))
        print(f"Image selected: {os.path.basename(os.path.normpath(filename2))}")

        print(f"Calculated error: {
        np.average(abs(
            np.array(img1, dtype="int16") -
            np.array(img2, dtype="int16"))
        )}")


if __name__ == "__main__":
    main()
