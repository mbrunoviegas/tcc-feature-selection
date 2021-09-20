import os
import tkinter as tk
from datetime import datetime
from math import pi
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askquestion, showerror, showinfo

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn import feature_selection, svm
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

from characteristics import Characteristics
from file import File
from results import Results
from selectCharacteristics import SelectCharacteristics
from featureSelection import FeatureSelection


class MainWindow():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Trabalho Prático")
        self.file = File()
        self.featureSelection = FeatureSelection()
        self.characteristics = Characteristics()
        self.selectCharacteristics = SelectCharacteristics()
        self.results = Results()
        self.initUi()

    def initUi(self):
        self.selecting = bool(0)
        self.scale = 1
        self.ret_id = 0
        self.selectedCharacteristics = [
            'homogeneity', 'contrast', 'energy', 'entropy', 'hu']
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
        # self.btn_read_directory = tk.Button(
            # self.fr_buttons, text="Read Directories", command=self.read_directory)
        # self.btn_characteristics = tk.Button(
            # self.fr_buttons, text="Characteristics", command=self.callCharacteristics)
        self.btn_classify = tk.Button(
            self.fr_buttons, text="Classify", command=self.classify_image)
        self.btn_train = tk.Button(
            self.fr_buttons, text="Train", command=self.trainSVM)
        # self.btn_select_characteristics = tk.Button(
            # self.fr_buttons, text="Select Characteristics", command=self.openSelectCharacteristics)

        self.btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_in.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_out.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_reset.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.btn_selection.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        # self.btn_read_directory.grid(
            # row=6, column=0, sticky="ew", padx=5, pady=5)
        # self.btn_characteristics.grid(
            # row=7, column=0, sticky="ew", padx=5, pady=5)
        self.btn_classify.grid(row=8, column=0, sticky="ew", padx=5, pady=5)
        self.btn_train.grid(row=9, column=0, sticky="ew", padx=5, pady=5)
        # self.btn_select_characteristics.grid(
            # row=10, column=0, sticky="ew", padx=5, pady=5)

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
        if (self.selecting):
            self.selection()
        self.image_original = (Image.open(self.filepath)).convert("L")


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

    # def callCharacteristics(self):
    #     self.inicio_characteristics = datetime.now()

    #     if (self.ret_id and self.selecting):
    #         imagePath = self.file.save_file(
    #             self.x_center, self.y_center, self.scale, self.image)
    #         # self.canvas.delete(self.ret_id)
    #     else:
    #         imagePath = self.filepath.split("images")[1]
    #         imagePath = "images/" + imagePath

    #     imagePath = imagePath.replace('dcm', 'png')
    #     # Converter para matriz RGB de 3 dimensões
    #     self.image_array = cv2.imread(imagePath)
    #     # converter para cinza
    #     self.image_array_gray = cv2.cvtColor(
    #         self.image_array, cv2.COLOR_BGR2GRAY)
    #     # converter para 32 tons de cinza
    #     self.image_array_gray = (self.image_array_gray / 8).astype('uint8')

    #     # Calcular caracteristicas
    #     self.characteristics.CalcShowCharacteristics(
    #         self.image_array_gray, self.inicio_characteristics)

    # def openSelectCharacteristics(self):
    #     def selectedCharacteristicsListCallBack(selectedCharacteristicsList):
    #         self.selectedCharacteristics = selectedCharacteristicsList
    #         print(self.selectedCharacteristics)

    #     _ = self.selectCharacteristics.showWindow(
    #         self.selectedCharacteristics, selectedCharacteristicsListCallBack)

    # def read_directory(self):
    #     self.images_class_1 = []
    #     self.images_class_2 = []
    #     self.images_class_3 = []
    #     self.images_class_4 = []

    #     self.images_class_1, self.images_class_2, self.images_class_3, self.images_class_4 = self.file.read_directory()

    # def train(self):
    #     self.train_time_start = datetime.now()

    #     if (self.images_class_1 == None and self.images_class_2 == None and self.images_class_3 == None and self.images_class_4 == None):
    #         print(
    #             "Diretórios ainda não carregados clique em 'Read Directories' para carrega-los")

    #     self.images_class_1_Characteristics = []
    #     self.images_class_2_Characteristics = []
    #     self.images_class_3_Characteristics = []
    #     self.images_class_4_Characteristics = []

    #     for img in self.images_class_1:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         img = (img/8).astype('uint8')
    #         img = self.characteristics.calcCharacteristics(
    #             img, self.selectedCharacteristics)
    #         self.images_class_1_Characteristics.append(img)

    #     for img in self.images_class_2:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         img = (img/8).astype('uint8')
    #         img = self.characteristics.calcCharacteristics(
    #             img, self.selectedCharacteristics)
    #         self.images_class_2_Characteristics.append(img)

    #     for img in self.images_class_3:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         img = (img/8).astype('uint8')
    #         img = self.characteristics.calcCharacteristics(
    #             img, self.selectedCharacteristics)
    #         self.images_class_3_Characteristics.append(img)

    #     for img in self.images_class_4:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         img = (img/8).astype('uint8')
    #         img = self.characteristics.calcCharacteristics(
    #             img, self.selectedCharacteristics)
    #         self.images_class_4_Characteristics.append(img)

    #     self.images_class_1_Characteristics = np.array(
    #         self.images_class_1_Characteristics)
    #     self.images_class_2_Characteristics = np.array(
    #         self.images_class_2_Characteristics)
    #     self.images_class_3_Characteristics = np.array(
    #         self.images_class_3_Characteristics)
    #     self.images_class_4_Characteristics = np.array(
    #         self.images_class_4_Characteristics)

    #     self.createLabel()

    # def createLabel(self):
    #     self.initLabels()
    #     self.initTrainTests()

    #     X_train = []
    #     X_test = []
    #     y_train = []
    #     y_test = []

    #     for i in range(0, 75):
    #         X_train.append(self.X_train1[i])
    #         X_train.append(self.X_train2[i])
    #         X_train.append(self.X_train3[i])
    #         X_train.append(self.X_train4[i])

    #         y_train.append(self.y_train1[i])
    #         y_train.append(self.y_train2[i])
    #         y_train.append(self.y_train3[i])
    #         y_train.append(self.y_train4[i])

    #     for i in range(0, 25):
    #         X_test.append(self.X_test1[i])
    #         X_test.append(self.X_test2[i])
    #         X_test.append(self.X_test3[i])
    #         X_test.append(self.X_test4[i])

    #         y_test.append(self.y_test1[i])
    #         y_test.append(self.y_test2[i])
    #         y_test.append(self.y_test3[i])
    #         y_test.append(self.y_test4[i])

    #     test_x = np.array(X_test)
    #     test_y = np.array(y_test)
    #     train_x = np.array(X_train)
    #     train_y = np.array(y_train)

    #     train_y = to_categorical(train_y, 4)
    #     self.initModel(test_x, test_y, train_x, train_y)
    #     self.classify_test(test_x, test_y)
        

    # def initLabels(self):
    #     self.label1 = np.full(100, 0)
    #     self.label2 = np.full(100, 1)
    #     self.label3 = np.full(100, 2)
    #     self.label4 = np.full(100, 3)

    # def initTrainTests(self):
    #     self.X_train1, self.X_test1, self.y_train1, self.y_test1 = train_test_split(
    #         self.images_class_1_Characteristics, self.label1, test_size=0.25, random_state=5)
    #     self.X_train2, self.X_test2, self.y_train2, self.y_test2 = train_test_split(
    #         self.images_class_2_Characteristics, self.label2, test_size=0.25, random_state=5)
    #     self.X_train3, self.X_test3, self.y_train3, self.y_test3 = train_test_split(
    #         self.images_class_3_Characteristics, self.label3, test_size=0.25, random_state=5)
    #     self.X_train4, self.X_test4, self.y_train4, self.y_test4 = train_test_split(
    #         self.images_class_4_Characteristics, self.label4, test_size=0.25, random_state=5)

    # def initModel(self, test_x, test_y, train_x, train_y):
    #   print(test_x.shape)
    #   print(test_y.shape)
    #   print(train_x.shape)
    #   print(train_y.shape)
    #   clf = svm.SVC(kernel='linear', probability=True, gamma="scale", C=1.0)
    #   print("Iniciando treinamento do SVM")
    #   clf.fit(train_x, train_y)
    #   infoString = ""
    #   y_predicted = clf.predict(test_x)
    #   print(f"Predicted Values {y_predicted}")
    #   print(f"Expected Values {test_y}")

    #   accuracy = accuracy_score(test_y, y_predicted)

    #   confusionMatrix = confusion_matrix(y_test, y_predicted)
    #   print(confusionMatrix)
    #   infoString += str(confusionMatrix)
    #   (mean_sensibility, specificity) = self.computeMetrics(confusionMatrix)
    #   print(f"Accuracy {accuracy}")
    #   infoString += f"\nAccuracy {accuracy}"
    #   infoString += f"\nSensiblidade Média: {mean_sensibility}"
    #   infoString += f"\nEspecificidade: {specificity}"

    # def classify_test(self, test_x, test_y):
    #     predictions = np.argmax(self.model.predict(test_x), axis=-1)
    #     confusion_matrix = self.calcConfusion(test_y, predictions)
    #     train_time_final = datetime.now()
    #     result_string = str(classification_report(test_y, predictions)) + '\n\n' + confusion_matrix + \
    #         '\n\n' + \
    #         f'Tempo de execução = {train_time_final - self.train_time_start}'

    #     self.results.show_results(result_string)

    #     print(result_string)

    # def calcConfusion(self, test_y, predictions):
    #     matrix = confusion_matrix(test_y, predictions)
    #     accuracy = self.calcAccuracy(matrix)
    #     specificity = self.calcSpecifiaty(matrix)

    #     return 'Matriz de confusão: ' + '\n' + f'{matrix}' + '\n' + 'Accuracy: ' + f'{accuracy}' + '\n' + 'Specificity: ' + f'{specificity}'

    # def calcAccuracy(self, matrix):
    #     matrix_sum = 0
    #     for i in range(4):
    #         matrix_sum += matrix[i][i]

    #     accuracy = matrix_sum / 100
    #     return accuracy

    # def calcSpecifiaty(self, matrix):
    #     specificity = 1
    #     matrix_sum = 0
    #     for i in range(4):
    #         for j in range(4):
    #             if i != j:
    #                 matrix_sum += matrix[i][j]

    #     specificity -= (matrix_sum / 300)
    #     return specificity

    # def countCharacteristics(self):
    #     counter = 0
    #     if ('homogeneity' in self.selectedCharacteristics):
    #         counter += 5

    #     if ('contrast' in self.selectedCharacteristics):
    #         counter += 5

    #     if ('energy' in self.selectedCharacteristics):
    #         counter += 5

    #     if ('entropy' in self.selectedCharacteristics):
    #         counter += 5

    #     if ('hu' in self.selectedCharacteristics):
    #         counter += 7

    #     return counter

    # def classify(self):
    #     if (self.ret_id and self.selecting):
    #         imagePath = self.file.save_file(
    #             self.x_center, self.y_center, self.scale, self.image)
    #         # self.canvas.delete(self.ret_id)
    #     else:
    #         imagePath = self.filepath.split("images")[1]
    #         imagePath = "images/" + imagePath

    #     imagePath = imagePath.replace('dcm', 'png')
    #     self.image_array = cv2.imread(imagePath)
    #     # converter para cinza
    #     self.image_array_gray = cv2.cvtColor(
    #         self.image_array, cv2.COLOR_BGR2GRAY)
    #     # converter para 32 tons de cinza
    #     self.image_array_gray = (self.image_array_gray/8).astype('uint8')

    #     # Calcular caracteristicas
    #     props = []
    #     props.append(self.characteristics.calcCharacteristics(
    #         self.image_array_gray, self.selectedCharacteristics))

    #     props = np.array(props)

    #     result_string = f'Classificada como classe: {np.argmax(self.model.predict(props), axis=-1) + 1}'

    #     self.results.show_results(result_string)

    #     print(result_string)

    def compute_entropy_for_glcm4D(self, glcm4D):
      entropyList = []
      for i in range(0, 5):
          for j in range(0, 4):
              entropy = shannon_entropy(glcm4D[:, :, i, j])
              entropyList.append(entropy)
      return entropyList


    def compute_descriptors(self, image):
        imageArray = np.array(image, dtype=np.uint8)
        glcm4D = greycomatrix(imageArray, distances=[1, 2, 4, 8, 16], angles=[0, pi / 4, pi / 2, 3 * pi / 4], levels=256)

        contrastMatrix = greycoprops(glcm4D, 'contrast')
        homogeneityMatrix = greycoprops(glcm4D, 'homogeneity')
        entropyList = self.compute_entropy_for_glcm4D(glcm4D)

        contrastList = np.hstack(contrastMatrix)
        homogeneityList = np.hstack(homogeneityMatrix)

        return list(contrastList) + list(homogeneityList) + list(entropyList)


    def compute_for_all_images_sizes(self, image):
        image128 = image.resize((128, 128))
        image64 = image.resize((64, 64))
        image32 = image.resize((32, 32))

        descriptors128 = self.compute_descriptors(image128)
        descriptors64 = self.compute_descriptors(image64)
        descriptors32 = self.compute_descriptors(image32)

        descriptors = descriptors32 + descriptors64 + descriptors128
        return descriptors


    def loadAndComputeDescriptorsAtPath(self, path=None, image=None):
        if image is None:
            image = Image.open(path)
        imageEqualized = ImageOps.equalize(image)
        imageGray = imageEqualized.convert("L")
        image16Colors = imageGray.quantize(colors=16)
        image32Colors = imageGray.quantize(colors=32)
        allDescriptors = self.compute_for_all_images_sizes(image32Colors) + self.compute_for_all_images_sizes(image16Colors)
        return allDescriptors


    def readImagesAndComputeDescriptors(self):
        basePath = "imgs/"
        types = []
        imagesDescriptors = []
        num_of_images_processed = 0
        for i in range(1, 5):
            for entry in os.scandir(basePath + str(i) + "/"):
                if entry.path.endswith(".png") and entry.is_file():
                    imagesDescriptors.append(self.loadAndComputeDescriptorsAtPath(entry.path))
                    types.append(i)
                    num_of_images_processed += 1
                    print(f"Gerando descritores {num_of_images_processed}/400")
        return imagesDescriptors, types


    def trainSVM(self):
        imagesDescriptors, types = self.readImagesAndComputeDescriptors()
        X_train, X_test, y_train, y_test = train_test_split(self.featureSelection.varianceTrashHold(imagesDescriptors),
                                                            types,
                                                            test_size=.25)

        self.clf = svm.SVC(kernel='linear', probability=True, gamma="scale", C=1.0)
        print("Iniciando treinamento do SVM")
        self.clf.fit(X_train, y_train)
        infoString = ""
        y_predicted = self.clf.predict(X_test)
        print(f"Predicted Values {y_predicted}")
        print(f"Expected Values {y_test}")

        accuracy = accuracy_score(y_test, y_predicted)

        confusionMatrix = confusion_matrix(y_test, y_predicted)

        print(confusionMatrix)
        infoString += str(confusionMatrix)
        (mean_sensibility, specificity) = self.computeMetrics(confusionMatrix)
        print(f"Accuracy {accuracy}")
        infoString += f"\nAccuracy {accuracy}"
        infoString += f"\nSensiblidade Média: {mean_sensibility}"
        infoString += f"\nEspecificidade: {specificity}"
        infoString += f"\Descritores: {imagesDescriptors}"
        # print(np.array(types).shape)
        # print(np.array(self.featureSelection.varianceTrashHold(imagesDescriptors)).shape)
        # print(np.array(imagesDescriptors).shape);
        return self.clf

    def computeMetrics(self, confusionMatrix):
        mean_sensibility = 0
        for i in range(0, 3):
            mean_sensibility += confusionMatrix[i][i] / 100

        sum = 0

        for i in range(0, 3):
            for j in range(0, 3):
                if i != j:
                    sum += confusionMatrix[i][j] / 300
        specificity = 1 - sum
        print(f"Sensiblidade Média: {mean_sensibility}")
        print(f"Especificidade: {specificity}")
        return (mean_sensibility, specificity)

    def open_image(self):
        self.filepath = askopenfilename(
            filetypes=[("Image Files", "*.png *.tif")]
        )
        if self.filepath == '':
            return
        self.image_original = (Image.open(self.filepath)).convert("L")
        self.image_on_screen = self.image_original
        self.photo_image = ImageTk.PhotoImage(self.image_on_screen)
        width, height = self.image_on_screen.size
        self.window.geometry(f"{width}x{height}")
        self.canvas.config(width=width, height=height)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def classify_image(self):
      if self.clf is None:
          print("CLF IS NONE")
          showerror("Error", "Não há nenhum classificador treinado")
          return
      if self.image_original is None:
          print("IMAGE IS NONE")
          showerror("Error", "Não há nenhuma imagem para ser classificada")
          return
      descriptors = self.loadAndComputeDescriptorsAtPath(image=self.image_original)
      descriptors = np.reshape(descriptors, (1, -1))
      predicted = self.clf.predict(descriptors)
      showinfo("Classificação", f"Birads {predicted[0]}")
      print(predicted)

def main():
    MainWindow()


main()
