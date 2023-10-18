import math

import customtkinter
import cv2
import numpy as np
from customtkinter import filedialog
from PIL import Image, ImageTk
import matplotlib
from matplotlib import pyplot as plt

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.geometry("500x150")


def fft_for_rotate(filepath):
    image = cv2.imread(filepath)
    # wyciąganie siatki i wykresu dla rotate
    image_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2HSV)

    # cv2.imshow('hsv', image_hsv)

    lower_range_red = np.array([140, 100, 100])
    upper_range_red = np.array([180, 255, 255])

    lower_range_black = np.array([0, 0, 0])
    upper_range_black = np.array([360, 255, 210])

    mask_grid = cv2.inRange(image_hsv, lower_range_red, upper_range_red)
    mask_black = cv2.inRange(image_hsv, lower_range_black, upper_range_black)

    fft_grid = np.fft.fft2(mask_grid)
    magnitude_grid = np.abs(fft_grid)
    normalized_magnitude_grid = cv2.normalize(magnitude_grid, None, 0, 255, cv2.NORM_MINMAX)
    normalized_magnitude_grid_uint8 = normalized_magnitude_grid.astype(np.uint8)

    return mask_grid, normalized_magnitude_grid_uint8


def rotate_chart(mask_grid, normalized_magnitude_grid_uint8, filepath):
    # rotating
    height, width = mask_grid.shape

    for x in range(width):
        for y in range(height):
            if normalized_magnitude_grid_uint8[y, x] > 50:
                normalized_magnitude_grid_uint8[y, x] = 255
            else:
                normalized_magnitude_grid_uint8[y, x] = 0

    pixels = np.argwhere(normalized_magnitude_grid_uint8 == 255)

    x1, y1 = pixels[2]
    x2, y2 = pixels[10]

    tan = (x1 - x2) / (y1 - y2)

    angle_radian = math.atan(tan)
    angle = math.degrees(angle_radian)

    rotate_img = Image.open(filepath)
    rotated_image = rotate_img.rotate(angle)

    return rotated_image


def change_colormap(rotated_image):
    rotated_image_rgb = rotated_image.convert("RGB")

    rotated_image_hsv = cv2.cvtColor(np.array(rotated_image_rgb), cv2.COLOR_RGB2HSV)

    lower_range_red = np.array([140, 100, 100])
    upper_range_red = np.array([180, 255, 255])

    lower_range_black = np.array([0, 0, 0])
    upper_range_black = np.array([360, 255, 210])

    mask_grid_chart = cv2.inRange(rotated_image_hsv, lower_range_red, upper_range_red)
    mask_chart = cv2.inRange(rotated_image_hsv, lower_range_black, upper_range_black)

    return mask_chart


def give_contours(mask_chart):
    # kontury
    cv2.imshow('Grayscale Image', mask_chart)
    ret, thresh = cv2.threshold(mask_chart, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask_chart, contours, -1, (0, 255, 0), 2)
    # print(contours)
    cv2.imshow("kontury", mask_chart)


def open_file():
    filepath = filedialog.askopenfilename()
    upload_image = ImageTk.PhotoImage(Image.open(filepath))
    # image_window = customtkinter.CTkToplevel(root)
    # image_window.title("Zapis EKG")
    # image_frame = customtkinter.CTkFrame(master=image_window)
    # image_frame.pack(fill="both", expand=True)
    # image_label = customtkinter.CTkLabel(master=image_frame, image=upload_image, text=" ")
    # image_label.pack(fill='both', expand=True)

    result = fft_for_rotate(filepath)
    rotated_image = rotate_chart(*result, filepath)
    mask_chart = change_colormap(rotated_image)
    give_contours(mask_chart)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Zapis EKG")
label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Wgraj obraz", command=lambda: open_file())
button.pack(pady=12, padx=10)

root.mainloop()
