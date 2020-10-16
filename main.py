import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename

class MainWindow():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Trabalho Prático")
        self.window.rowconfigure(0, minsize=800, weight=1)
        self.window.columnconfigure(1, minsize=800, weight=1)
        self.initUi()

    def initUi(self):
        self.selecionando = bool(0)
        self.ret_id = 0
        self.scale = 1
        self.btn_selecionar_text = tk.StringVar()
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

        self.btn_selecionar_text.set('Selecionar Área')

        self.btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_in.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_out.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_reset.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.btn_selecionar.grid(row=5, column=0, sticky="ew", padx=5, pady=5)

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
        if (self.ret_id != 0):
            pos_x1 = self.x_centro - 64 * (self.scale)
            pos_y1 = self.y_centro + 64 * (self.scale)
            pos_x2 = self.x_centro + 64 * (self.scale)
            pos_y2 = self.y_centro - 64 * (self.scale)
            self.ret_id = self.canvas.create_rectangle(pos_x1, pos_y1, pos_x2, pos_y2,outline="green")

    def selecionar(self):
        if (self.selecionando):
            self.selecionando = bool(0)
            self.btn_selecionar_text.set('Selecionar Área')
        else:
            self.selecionando = bool(1)
            self.btn_selecionar_text.set('Cancelar Seleção')

    def click(self, event):
        self.canvas.scan_mark(event.x, event.y)
        if(self.ret_id):
            self.canvas.delete(self.ret_id)

        if self.selecionando:
            pos_x1 = event.x - 64 * (self.scale)
            pos_y1 = event.y + 64 * (self.scale)
            pos_x2 = event.x + 64 * (self.scale)
            pos_y2 = event.y - 64 * (self.scale)
            
            if pos_x1 > 0 and pos_x2 < self.canvas.winfo_width() and pos_y1 > 0 and pos_y2 < self.canvas.winfo_height():
                self.ret_id = self.canvas.create_rectangle(pos_x1, pos_y1, pos_x2, pos_y2,outline="green")
                self.x_centro = event.x
                self.y_centro = event.y

    def drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


def main():
    MainWindow()


main()