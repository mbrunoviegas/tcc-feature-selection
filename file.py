import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from os import listdir
from os.path import isfile, join
import SimpleITK as sitk


class File():
    def open_file(self):
        self.filepath = askopenfilename(
            filetypes=[("Image Files", "*.png *.tiff *.dcm"),
                       ("All Files", "*.*")]
        )

        if(self.filepath):
            self.file_type = self.filepath.split('.')[-1]
            if (self.file_type == 'dcm'):
                self.open_dicom(self.filepath)
            else:
                self.open_image(self.filepath)
            self.response = self.image, self.photo, self.filepath
            return self.response
        else:
            return (bool(0), bool(0), bool(0))

    def open_image(self, imgPath):
        self.image = Image.open(imgPath)
        self.width, self.height = self.image.size
        self.photo = ImageTk.PhotoImage(self.image)

    def open_dicom(self, filepath):
        self.png_name = filepath.replace('.dcm', '.png')
        self.convert_image(self.filepath, self.png_name)
        self.open_image(self.png_name)

    def read_directory(self):
        onlyfiles1 = [f for f in listdir(
            "images/1") if isfile(join("images/1", f))]
        print("--------- Diretorio 1 ---------\n", onlyfiles1)
        onlyfiles2 = [f for f in listdir(
            "images/2") if isfile(join("images/2", f))]
        print("\n--------- Diretorio 2 ---------\n", onlyfiles2)
        onlyfiles3 = [f for f in listdir(
            "images/3") if isfile(join("images/3", f))]
        print("\n--------- Diretorio 3 ---------\n", onlyfiles3)
        onlyfiles4 = [f for f in listdir(
            "images/4") if isfile(join("images/4", f))]
        print("\n--------- Diretorio 4 ---------\n", onlyfiles4)

    def convert_image(self, input_file_name, output_file_name):
        try:
            image_file_reader = sitk.ImageFileReader()
            image_file_reader.SetImageIO('GDCMImageIO')
            image_file_reader.SetFileName(input_file_name)
            image_file_reader.ReadImageInformation()
            image_size = list(image_file_reader.GetSize())

            if len(image_size) == 3 and image_size[2] == 1:
                image_size[2] = 0

            image_file_reader.SetExtractSize(image_size)
            image = image_file_reader.Execute()

            if image.GetNumberOfComponentsPerPixel() == 1:
                image = sitk.RescaleIntensity(image, 0, 255)
                if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                    image = sitk.InvertIntensity(image, maximum=255)
                image = sitk.Cast(image, sitk.sitkUInt8)

            sitk.WriteImage(image, output_file_name)
            return True
        except BaseException:
            return False

    def save_file(self, x_center, y_center, scale, image):
        x = x_center/scale - 64
        y = y_center/scale - 64
        cropped = image.crop((x, y, x + 128, y + 128))
        cropped.load()
        cropped_name = self.filepath.split('/')[-1]
        cropped_name = cropped_name.split('.')[0]
        cropped.save(cropped_name + ".png", "PNG")

    def resize_image(self, scale):
        self.size = int(self.width * scale), int(self.height * scale)
        self.width_resize, self.height_resize = self.size
        return self.width_resize, self.height_resize, ImageTk.PhotoImage(self.image.resize(self.size))