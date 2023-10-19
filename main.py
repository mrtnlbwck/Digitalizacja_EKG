import math

import PIL
import customtkinter
import cv2
import numpy as np
from customtkinter import filedialog
from PIL import Image, ImageTk, ImageOps
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

    # cv2.imshow('grid', mask_grid)

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


def color_change_chart(rotated_image):
    rotated_image_rgb = rotated_image.convert("RGB")

    img_gray = cv2.cvtColor(np.array(rotated_image_rgb), cv2.COLOR_RGB2GRAY)

    return img_gray


def gaussian_filter(img_gray):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret3, chart = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow('chart', chart)

    return chart


def remove_isolated_pixels(chart_after_filter_1):
    chart_after_filter_1_pil = Image.fromarray(np.uint8(chart_after_filter_1))

    # Invert the colors of the PIL image
    chart_after_filter_np = PIL.ImageOps.invert(chart_after_filter_1_pil)

    # print (type(chart_after_filter_np))
    chart_after_filter = np.array(chart_after_filter_np)
    # load image, ensure binary, remove bar on the left

    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(chart_after_filter, kernel, iterations=1)
    # cv2.imshow('erosion', img_erosion)

    img_erosion = cv2.threshold(img_erosion, 254, 255, cv2.THRESH_BINARY)[1]
    input_image_comp = cv2.bitwise_not(img_erosion)  # could just use 255-img

    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)

    hitormiss1 = cv2.morphologyEx(img_erosion, cv2.MORPH_ERODE, kernel1)
    hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
    hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)

    # cv2.imshow('isolated', hitormiss)

    hitormiss_comp = cv2.bitwise_not(hitormiss)  # could just use 255-img
    del_isolated = cv2.bitwise_and(img_erosion, img_erosion, mask=hitormiss_comp)
    cv2.imshow('removed', del_isolated)

    return del_isolated


def put_into_pixels(chart_after_filter):
    pixels = np.argwhere(chart_after_filter == 255)

    x = pixels[:, 1]
    y = - pixels[:, 0]

    print(x)
    print(y)
    print(pixels)

    # Wykres punktowy
    plt.scatter(x, y, marker='o', color='b', label='Punkty')

    # Dodatkowe opcje konfiguracyjne
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.title('Wykres punktów')
    plt.legend()  # Dodaj legendę, jeśli chcesz

    plt.show()

    print(pixels)

    return pixels


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
    img_gray = color_change_chart(rotated_image)
    chart_after_filter = gaussian_filter(img_gray)
    chart_after_delete = remove_isolated_pixels(chart_after_filter)
    put_into_pixels(chart_after_delete)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Zapis EKG")
label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Wgraj obraz", command=lambda: open_file())
button.pack(pady=12, padx=10)

root.mainloop()
