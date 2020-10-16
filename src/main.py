import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from os import listdir
from os.path import isfile, join

# TODO Arrumar o cancelar selecção
# TODO Abrir diretórios
# TODO Arrumar o zoom
# TODO Arrumar referencia ao mexer a imagem
# TODO Colocar os botões que faltam

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
        self.window.rowconfigure(0, minsize=800, weight=1)
        self.window.columnconfigure(1, minsize=800, weight=1)
        self.initUi()

    def initUi(self):
        self.selecionando = bool(0)
        self.scale = 1
        self.ret_id = 0
        self.btn_selecionar_text = tk.StringVar()
        self.btn_selecionar_text.set('Selection')
        self.canvas = Canvas(width=500, height=500, bg='white')


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
        self.btn_selecionar = tk.Button(
            self.fr_buttons, textvariable=self.btn_selecionar_text, command=self.selecionar)
        self.btn_read_directory = tk.Button(
            self.fr_buttons, text="Read Directories", command=self.read_directory)
        self.btn_characteristics= tk.Button(
            self.fr_buttons, text="Characteristics", command=self.characteristics)
        self.btn_classify= tk.Button(
            self.fr_buttons, text="Classify", command=self.classify)

        self.btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_in.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_out.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_reset.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.btn_selecionar.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        self.btn_read_directory.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        self.btn_characteristics.grid(row=7, column=0, sticky="ew", padx=5, pady=5)
        self.btn_classify.grid(row=8, column=0, sticky="ew", padx=5, pady=5)

        self.fr_buttons.grid(row=0, column=0, sticky="ns")
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<Button-1>', self.click)

        self.window.mainloop()

    def open_file(self):
        self.filepath = askopenfilename(
            filetypes=[("Image Files", "*.png *.tiff *.dicom *.dcm"),
                       ("All Files", "*.*")]
        )
        self.image = Image.open(self.filepath)
        self.photo = ImageTk.PhotoImage(self.image)
        self.id_img = self.canvas.create_image(250, 250, image=self.photo)
        self.window.title(f"Trabalho Prático - {self.filepath}")
        self.zoom_reset()

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
        iw, ih = self.image.size
        size = int(iw * self.scale), int(ih * self.scale)
        self.photo = ImageTk.PhotoImage(self.image.resize(size))
        self.canvas.create_image(250, 250, image=self.photo)
        if (self.ret_id):
            self.canvas.delete(self.ret_id)
            if(self.selecionando):
                self.drawRectangle(self.x_center, self.y_center)

    def selecionar(self):
        if (self.selecionando):
            self.selecionando = bool(0)
            self.btn_selecionar_text.set('Seletion')
        else:
            self.selecionando = bool(1)
            self.btn_selecionar_text.set('Cancel Selection')

    def click(self, event):
        self.canvas.scan_mark(event.x, event.y)
        if(self.ret_id):
            self.canvas.delete(self.ret_id)

        if self.selecionando:
            self.x_center = event.x
            self.y_center = event.y
            self.drawRectangle(event.x, event.y)

    def drawRectangle(self, x, y):
        pos_x1 = x - 64 * (self.scale)
        pos_y1 = y + 64 * (self.scale)
        pos_x2 = x + 64 * (self.scale)
        pos_y2 = y - 64 * (self.scale)

        if pos_x1 > 0 and pos_x2 < self.canvas.winfo_width() and pos_y1 > 0 and pos_y2 < self.canvas.winfo_height():
            self.ret_id = self.canvas.create_rectangle(
                pos_x1, pos_y1, pos_x2, pos_y2, outline="green")

    def drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def read_directory(self):
        self.onlyfiles1 = [f for f in listdir("../images/1") if isfile(join("../images/1", f))]
        print("--------- Diretorio 1 ---------\n", self.onlyfiles1)
        self.onlyfiles2 = [f for f in listdir("../images/2") if isfile(join("../images/2", f))]
        print("\n--------- Diretorio 2 ---------\n", self.onlyfiles2)
        self.onlyfiles3 = [f for f in listdir("../images/3") if isfile(join("../images/3", f))]
        print("\n--------- Diretorio 3 ---------\n", self.onlyfiles3)
        self.onlyfiles4 = [f for f in listdir("../images/4") if isfile(join("../images/4", f))]
        print("\n--------- Diretorio 4 ---------\n", self.onlyfiles4)

    def characteristics(self):
        pass

    def classify(self):
        pass


def main():
    MainWindow()


main()
