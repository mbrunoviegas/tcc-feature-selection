import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from os import listdir
from os.path import isfile, join
import SimpleITK as sitk

# TODO Arrumar o cancelar selecção
# TODO Colocar  CROP

# (a) ler o diretório de imagens de treino/teste;
# (b) selecionar as características a serem usadas; ---- Botao Criado
# (c) treinar o classificador;  ---- Botao Criado
# (d) abrir e visualizar uma imagem; ---- FEITO
# (e) marcar a região de interesse da imagem visualizada com o mouse; ------- FEITO
#  (f) calcular e exibir as características para a imagem visualizada ou área selecionada;
#  (g) classificar a imagem ou a região de interesse selecionada com o mouse.


class MainWindow():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Trabalho Prático")
        self.initUi()

    def initUi(self):
        self.selecionando = bool(0)
        self.scale = 1
        self.ret_id = 0
        self.btn_selection_text = tk.StringVar()
        self.btn_selection_text.set('Selection')

        self.fr_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)
        self.btn_open = tk.Button(
            self.fr_buttons, text="Open", command=self.open_file)
        self.btn_save = tk.Button(
            self.fr_buttons, text="Save As...", command=self.save_file)
        self.btn_zoom_in = tk.Button(
            self.fr_buttons, text="Zoom in", command=self.zoom_in)
        self.btn_zoom_out = tk.Button(
            self.fr_buttons, text="Zoom out", command=self.zoom_out)
        self.btn_zoom_reset = tk.Button(
            self.fr_buttons, text="Reset Zoom", command=self.zoom_reset)
        self.btn_selection = tk.Button(
            self.fr_buttons, textvariable=self.btn_selection_text, command=self.selection)
        self.btn_read_directory = tk.Button(
            self.fr_buttons, text="Read Directories", command=self.read_directory)
        self.btn_characteristics = tk.Button(
            self.fr_buttons, text="Characteristics", command=self.characteristics)
        self.btn_classify = tk.Button(
            self.fr_buttons, text="Classify", command=self.classify)

        self.btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_in.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_out.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_reset.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.btn_selection.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        self.btn_read_directory.grid(
            row=6, column=0, sticky="ew", padx=5, pady=5)
        self.btn_characteristics.grid(
            row=7, column=0, sticky="ew", padx=5, pady=5)
        self.btn_classify.grid(row=8, column=0, sticky="ew", padx=5, pady=5)

        self.fr_buttons.grid(row=0, column=0, sticky="ns")

        self.window.mainloop()

    def createCanvas(self, width, height):
        self.fr_canvas = tk.Frame(self.window, relief=tk.RAISED, bd=2)
        self.fr_canvas.grid(row=0, column=1, sticky="nsew")
        scroll_area = width, height
        if (width > 1080):
            width = 1080
        if (height > 600):
            height = 600
        self.canvas = Canvas(self.fr_canvas, width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.fr_canvas.grid_columnconfigure(0, weight=1)
        self.fr_canvas.grid_rowconfigure(0, weight=1)
        self.scrollbarY = Scrollbar(
            self.fr_canvas, orient="vertical", command=self.canvas.yview)
        self.scrollbarX = Scrollbar(
            self.fr_canvas, orient="horizontal", command=self.canvas.xview)
        self.scrollbarY.grid(row=0, column=1, sticky='ns')
        self.scrollbarX.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(yscrollcommand=self.scrollbarY.set,
                              xscrollcommand=self.scrollbarX.set)
        self.canvas.configure(scrollregion=(
            0, 0, scroll_area[0], scroll_area[1]))

        self.canvas.bind('<Button-1>', self.click)

    def open_file(self):
        self.filepath = askopenfilename(
            filetypes=[("Image Files", "*.png *.tiff *.dcm"),
                       ("All Files", "*.*")]
        )
        file_type = self.filepath.split('.')[-1]
        if (file_type == 'dcm'):
            self.openDicom(self.filepath)
        else:
            self.openImage(self.filepath)

    def openImage(self, imgPath):
        self.image = Image.open(imgPath)
        self.width, self.height = self.image.size
        self.photo = ImageTk.PhotoImage(self.image)
        self.createCanvas(self.width, self.height)
        self.id_img = self.canvas.create_image(
            0, 0, image=self.photo, anchor=NW)
        self.window.title(f"Trabalho Prático - {imgPath}")
        self.zoom_reset()

    def openDicom(self, filepath):
        self.png_name = self.filepath.replace('.dcm', '.png')
        self.convert_image(filepath, self.png_name)
        self.openImage(self.png_name)

    def save_file(self):
        pass

    def zoom_in(self):
        self.scale = self.scale*1.5
        self.redraw()
        self.btn_zoom_out['state'] = ACTIVE
        if (self.scale > 6):
            self.btn_zoom_in['state'] = DISABLED

    def zoom_out(self):
        self.scale = self.scale/1.5
        self.redraw()
        self.btn_zoom_in['state'] = ACTIVE
        if (self.scale < 0.025):
            self.btn_zoom_out['state'] = DISABLED

    def zoom_reset(self):
        self.btn_zoom_in['state'] = ACTIVE
        self.btn_zoom_out['state'] = ACTIVE
        self.scale = 1
        self.redraw()

    def redraw(self):
        if self.photo:
            self.canvas.delete(self.photo)
        self.size = int(self.width * self.scale), int(self.height * self.scale)
        self.photo = ImageTk.PhotoImage(self.image.resize(self.size))
        self.width_resize, self.height_resize = self.size
        self.createCanvas(self.width_resize, self.height_resize)
        self.canvas.create_image(0, 0, image=self.photo,  anchor=NW)
        if (self.ret_id):
            self.canvas.delete(self.ret_id)
            if(self.selecionando):
                self.drawRectangle(self.x_center, self.y_center)

    def selection(self):
        if (self.selecionando):
            self.selecionando = bool(0)
            self.btn_selection_text.set('Seletion')
        else:
            self.selecionando = bool(1)
            self.btn_selection_text.set('Cancel Selection')

    def click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if(self.ret_id):
            self.canvas.delete(self.ret_id)

        if self.selecionando:
            self.x_center = x
            self.y_center = y
            self.drawRectangle(x, y)

    def drawRectangle(self, x, y):
        pos_x1 = x * (self.scale) - 64 * (self.scale)
        pos_y1 = y * (self.scale) + 64 * (self.scale)
        pos_x2 = x * (self.scale) + 64 * (self.scale)
        pos_y2 = y * (self.scale) - 64 * (self.scale)

        self.ret_id = self.canvas.create_rectangle(
            pos_x1, pos_y1, pos_x2, pos_y2, outline="green")

    def drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def read_directory(self):
        self.onlyfiles1 = [f for f in listdir(
            "images/1") if isfile(join("images/1", f))]
        print("--------- Diretorio 1 ---------\n", self.onlyfiles1)
        self.onlyfiles2 = [f for f in listdir(
            "images/2") if isfile(join("images/2", f))]
        print("\n--------- Diretorio 2 ---------\n", self.onlyfiles2)
        self.onlyfiles3 = [f for f in listdir(
            "images/3") if isfile(join("images/3", f))]
        print("\n--------- Diretorio 3 ---------\n", self.onlyfiles3)
        self.onlyfiles4 = [f for f in listdir(
            "images/4") if isfile(join("images/4", f))]
        print("\n--------- Diretorio 4 ---------\n", self.onlyfiles4)

    def characteristics(self):
        pass

    def classify(self):
        pass

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

def main():
    MainWindow()

main()