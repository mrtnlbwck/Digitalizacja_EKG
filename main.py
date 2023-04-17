import sys
from tkinter import Toplevel

import customtkinter
import numpy
from customtkinter import filedialog
import PIL
from PIL import Image
import os

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.geometry("500x150")


def open_file():
    filepath = filedialog.askopenfilename()
    upload_image = customtkinter.CTkImage(Image.open(filepath), size=(702, 496))
    image_window = customtkinter.CTkToplevel(root)
    image_window.title("Zapis EKG")
    image_frame = customtkinter.CTkFrame(master=image_window)
    image_frame.pack()
    image_label = customtkinter.CTkLabel(master=image_frame, image=upload_image, text=" ")
    image_label.pack()
    upload_image.create_scaled_photo_image(1, "system")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Zapis EKG")
label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Wgraj obraz", command=lambda: open_file())
button.pack(pady=12, padx=10)

root.mainloop()
