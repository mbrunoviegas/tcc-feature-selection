from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(200, 200, 700, 500)
        self.setWindowTitle('Trabalho Prático')
        self.setWindowIcon(QtGui.QIcon("1043385.png"))
        self.initUi()

    def initUi(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText('My second label')
        self.label.move(50, 50)

        self.buttonSelecionarImagem = QtWidgets.QPushButton(self)
        self.buttonSelecionarImagem.move(580, 450)
        self.buttonSelecionarImagem.setText('Selecionar Imagem')
        self.buttonSelecionarImagem.clicked.connect(self.click_ButtonSelecionarImagem)

    def click_ButtonSelecionarImagem(self):
        self.winImage = ImageWindow()
        self.winImage.show()


class ImageWindow(QMainWindow):
    def __init__(self):
        super(ImageWindow, self).__init__()
        self.setGeometry(200, 200, 300, 300)
        self.setWindowTitle('Imagem selecionada')
        self.initUi()

    def initUi(self):
        self.buttonZoom = QtWidgets.QPushButton(self)
        self.buttonZoom.move(10, 10)
        self.buttonZoom.setText('Zoom')
        #self.buttonZoom.clicked.connect(self.click_ButtonSelecionarImagem)

        self.image = QtWidgets.QLabel(self)
        self.fname = QFileDialog.getOpenFileName( self, 'Open file','c:\\', "Image files (*.png *.tiff *.dicom *.dcm)" )
        self.imagePath = self.fname[0]
        self.pixmap = QPixmap(self.imagePath)
        self.image.setPixmap(QPixmap(self.pixmap))
        self.setCentralWidget(self.image)
        self.resize(1080, 720)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()
    sys.exit(app.exec_())


main()
