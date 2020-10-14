import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename, asksaveasfilename

class MainWindow():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Trabalho Prático")
        self.window.rowconfigure(0, minsize=800, weight=1)
        self.window.columnconfigure(1, minsize=800, weight=1)
        self.initUi()

    def initUi(self):
        self.canvas = Canvas(width=500, height=500, bg='black')

        self.fr_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)
        self.btn_open = tk.Button(self.fr_buttons, text="Open", command=self.open_file)
        self.btn_save = tk.Button(self.fr_buttons, text="Save As...", command=self.save_file)
        self.btn_zoom_out = tk.Button(self.fr_buttons, text="Zoom out", command=self.zoom_out)
        self.btn_zoom_in = tk.Button(self.fr_buttons, text="Zoom out", command=self.zoom_in)
        self.btn_selecionar = tk.Button(self.fr_buttons, text="Selecionar Área", command=self.selecionar)

        self.btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_out.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_in.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn_selecionar.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.fr_buttons.grid(row=0, column=0, sticky="ns")
        self.canvas.grid(row=0, column=1, sticky="nsew")

        self.window.mainloop()

    def open_file(self):
        filepath = askopenfilename(
            filetypes=[("Image Files", "*.png *.tiff *.dicom *.dcm"), ("All Files", "*.*")]
        )
        self.image = Image.open(filepath)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(250, 250, image=self.photo)
        self.window.title(f"Trabalho Prático - {filepath}")

    def save_file(self):
        pass

    def zoom_in(self):
        pass

    def zoom_out(self):
        pass

    def selecionar(self):
        pass


def main():
    MainWindow()

main()

# from PyQt5 import QtWidgets, QtGui
# from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
# from PyQt5.QtGui import QPixmap
# import sys
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.setGeometry(200, 200, 700, 500)
#         self.setWindowTitle('Trabalho Prático')
#         self.setWindowIcon(QtGui.QIcon("1043385.png"))
#         self.initUi()
#
#     def initUi(self):
#         self.label = QtWidgets.QLabel(self)
#         self.label.setText('My second label')
#         self.label.move(50, 50)
#
#         self.buttonSelecionarImagem = QtWidgets.QPushButton(self)
#         self.buttonSelecionarImagem.move(580, 450)
#         self.buttonSelecionarImagem.setText('Selecionar Imagem')
#         self.buttonSelecionarImagem.clicked.connect(self.click_ButtonSelecionarImagem)
#
#     def click_ButtonSelecionarImagem(self):
#         self.winImage = ImageWindow()
#         self.winImage.show()
#
#
# class ImageWindow(QMainWindow):
#     def __init__(self):
#         super(ImageWindow, self).__init__()
#         self.setGeometry(200, 200, 300, 300)
#         self.setWindowTitle('Imagem selecionada')
#         self.initUi()
#
#     def initUi(self):
#         self.buttonZoom = QtWidgets.QPushButton(self)
#         self.buttonZoom.move(10, 10)
#         self.buttonZoom.setText('Zoom')
#         #self.buttonZoom.clicked.connect(self.click_ButtonSelecionarImagem)
#
#         self.image = QtWidgets.QLabel(self)
#         self.fname = QFileDialog.getOpenFileName( self, 'Open file','c:\\', "Image files (*.png *.tiff *.dicom *.dcm)" )
#         self.imagePath = self.fname[0]
#         self.pixmap = QPixmap(self.imagePath)
#         self.image.setPixmap(QPixmap(self.pixmap))
#         self.setCentralWidget(self.image)
#         self.resize(1080, 720)
#
#
# def main():
#     app = QApplication(sys.argv)
#     win = MainWindow()
#
#     win.show()
#     sys.exit(app.exec_())
# main()
# ----------------------------------------------------------------------------------------------------
# def open_file():
#     filepath = askopenfilename(
#         filetypes=[("Image Files", "*.png *.tiff *.dicom *.dcm"), ("All Files", "*.*")]
#     )
#     image = Image.open(filepath)
#     photo = ImageTk.PhotoImage(image)
#     canvas.create_image(250, 250, image=photo)
#     window.title(f"Trabalho Prático - {filepath}")
#
# def save_file():
#     pass
#
# def zoom_in():
#     pass
#
# def zoom_out():
#     pass
#
# def selecionar():
#     pass


# window = tk.Tk()
# window.title("Trabalho Prático")
# window.rowconfigure(0, minsize=800, weight=1)
# window.columnconfigure(1, minsize=800, weight=1)
#
# canvas = Canvas(width=500, height=500, bg='black')
#
# fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
# btn_open = tk.Button(fr_buttons, text="Open", command=open_file)
# btn_save = tk.Button(fr_buttons, text="Save As...", command=save_file)
# btn_zoom_out = tk.Button(fr_buttons, text="Zoom out", command=zoom_out)
# btn_zoom_in = tk.Button(fr_buttons, text="Zoom out", command=zoom_in)
# btn_selecionar = tk.Button(fr_buttons, text="Selecionar Área", command=selecionar)
#
# btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
# btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
# btn_zoom_out.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
# btn_zoom_in.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
# btn_selecionar.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
#
# fr_buttons.grid(row=0, column=0, sticky="ns")
# canvas.grid(row=0, column=1, sticky="nsew")
# filepath = askopenfilename(
#     filetypes=[("Image Files", "*.png *.tiff *.dicom *.dcm"), ("All Files", "*.*")]
# )
# image = Image.open(filepath)
# photo = ImageTk.PhotoImage(image)
# canvas.create_image(250, 250, image=photo)
# window.title(f"Trabalho Prático - {filepath}")

# window.mainloop()