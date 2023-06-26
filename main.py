import customtkinter
import cv2
import numpy as np
from customtkinter import filedialog
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.geometry("500x150")


def open_file():
    filepath = filedialog.askopenfilename()
    upload_image = ImageTk.PhotoImage(Image.open(filepath))
    image_window = customtkinter.CTkToplevel(root)
    image_window.title("Zapis EKG")
    image_frame = customtkinter.CTkFrame(master=image_window)
    image_frame.pack(fill="both", expand=True)
    image_label = customtkinter.CTkLabel(master=image_frame, image=upload_image, text=" ")
    image_label.pack(fill='both', expand=True)

    #image = Image.open(filepath)
    image = cv2.imread(filepath)

    image_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2HSV)

    lower_range = np.array([169, 50, 50])
    upper_range = np.array([189, 255, 255])

    mask = cv2.inRange(image_hsv, lower_range, upper_range)

    cv2.imshow("Image", np.array(image))
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Zapis EKG")
label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Wgraj obraz", command=lambda: open_file())
button.pack(pady=12, padx=10)

root.mainloop()

