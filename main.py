import tkinter as tk
from tkinter import *
from file import File

class MainWindow():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Trabalho Prático")
        self.file = File()
        self.initUi()

    def initUi(self):
        self.selecting = bool(0)
        self.scale = 1
        self.ret_id = 0
        self.btn_selection_text = tk.StringVar()
        self.btn_selection_text.set('Selection')

        self.fr_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)
        self.btn_open = tk.Button(
            self.fr_buttons, text="Open", command=self.openFile)
        self.btn_save = tk.Button(
            self.fr_buttons, text="Save Selection", command=self.saveFile)
        self.btn_zoom_in = tk.Button(
            self.fr_buttons, text="Zoom in", command=self.zoom_in)
        self.btn_zoom_out = tk.Button(
            self.fr_buttons, text="Zoom out", command=self.zoom_out)
        self.btn_zoom_reset = tk.Button(
            self.fr_buttons, text="Reset Zoom", command=self.zoom_reset)
        self.btn_selection = tk.Button(
            self.fr_buttons, textvariable=self.btn_selection_text, command=self.selection)
        self.btn_read_directory = tk.Button(
            self.fr_buttons, text="Read Directories", command=self.file.read_directory)
        self.btn_characteristics = tk.Button(
            self.fr_buttons, text="Characteristics", command=self.characteristics)
        self.btn_classify = tk.Button(
            self.fr_buttons, text="Classify", command=self.classify)
        self.btn_train = tk.Button(
            self.fr_buttons, text="Train", command=self.classify)

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
        self.btn_train.grid(row=9, column=0, sticky="ew", padx=5, pady=5)

        self.fr_buttons.grid(row=0, column=0, sticky="ns")
        self.window.mainloop()

    def createCanvas(self, width, height):
        self.createFrameCanvas()
        self.scroll_area = width, height

        if (width > 1080):
            width = 1080
        if (height > 600):
            height = 600

        self.canvas = Canvas(self.fr_canvas, width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.createCanvasScrollbar()

        self.canvas.bind('<Button-1>', self.click)

    def createFrameCanvas(self):
        self.fr_canvas = tk.Frame(self.window, relief=tk.RAISED, bd=2)
        self.fr_canvas.grid(row=0, column=1, sticky="nsew")
        self.fr_canvas.grid_columnconfigure(0, weight=1)
        self.fr_canvas.grid_rowconfigure(0, weight=1)

    def createCanvasScrollbar(self):
        self.scrollbarY = Scrollbar(
            self.fr_canvas, orient="vertical", command=self.canvas.yview)
        self.scrollbarX = Scrollbar(
            self.fr_canvas, orient="horizontal", command=self.canvas.xview)
        self.scrollbarY.grid(row=0, column=1, sticky='ns')
        self.scrollbarX.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(yscrollcommand=self.scrollbarY.set,
                              xscrollcommand=self.scrollbarX.set,
                              scrollregion=(
                                  0, 0, self.scroll_area[0], self.scroll_area[1])
                              )

    def openFile(self):
        self.image, self.photo, self.filepath = self.file.open_file()
        if(self.image and self.photo and self.filepath):
            self.width, self.height = self.image.size
            self.createCanvas(self.width, self.height)
            self.id_img = self.canvas.create_image(
                0, 0, image=self.photo, anchor=NW)
            self.window.title(f"Trabalho Prático - {self.filepath}")
            self.zoom_reset()

    def saveFile(self):
        if (self.ret_id and self.selecting):
            self.file.save_file(
                self.x_center, self.y_center, self.scale, self.image)
            print("Imagem salva")
        else:
            print("Selecione uma área na imagem carregada")

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

        self.width_resize, self.height_resize, self.photo = self.file.resize_image(
            self.scale)
        self.createCanvas(self.width_resize, self.height_resize)
        self.canvas.create_image(0, 0, image=self.photo,  anchor=NW)

        if (self.ret_id):
            self.canvas.delete(self.ret_id)

    def selection(self):
        if (self.selecting):
            self.selecting = bool(0)
            self.btn_selection_text.set('Seletion')
            if(self.ret_id):
                self.canvas.delete(self.ret_id)
        else:
            self.selecting = bool(1)
            self.btn_selection_text.set('Cancel Selection')

    def click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if(self.ret_id):
            self.canvas.delete(self.ret_id)

        if self.selecting:
            self.x_center = x
            self.y_center = y
            self.drawRectangle(x, y)

    def drawRectangle(self, x, y):
        pos_x1 = x - 64 * (self.scale)
        pos_y1 = y + 64 * (self.scale)
        pos_x2 = x + 64 * (self.scale)
        pos_y2 = y - 64 * (self.scale)

        self.ret_id = self.canvas.create_rectangle(
            pos_x1, pos_y1, pos_x2, pos_y2, outline="green")

    def drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def characteristics(self):
        pass

    def classify(self):
        pass

    def train(self):
        pass


def main():
    MainWindow()

main()