import customtkinter
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

    img = Image.open(filepath)
    image_rgb = img.convert("RGB")
    red, green, blue = image_rgb.split()

    red_img = Image.merge("RGB", (red, red, red))
    green_img = Image.merge("RGB", (green, green, green))
    blue_img = Image.merge("RGB", (blue, blue, blue))

    red_img.save("red.jpg")
    green_img.save("green.jpg")
    blue_img.save("blue.jpg")


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Zapis EKG")
label.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Wgraj obraz", command=lambda: open_file())
button.pack(pady=12, padx=10)

root.mainloop()

