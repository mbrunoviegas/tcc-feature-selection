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
from skimage.measure import (moments_central, moments_hu, moments_normalized,
                             shannon_entropy)
from sklearn import feature_selection, svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from characteristics import Characteristics
from featureSelection import (correlationBasedFeature,
                              univariateFeatureSelection, varianceTrashHold)
from file import File
from results import Results
from selectCharacteristics import SelectCharacteristics


class MainWindow():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Trabalho Prático")
        self.file = File()
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
        self.btn_classify = tk.Button(
            self.fr_buttons, text="Classify", command=self.classify_image)
        self.btn_train = tk.Button(
            self.fr_buttons, text="Train", command=self.trainSVM)
        self.btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_in.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_out.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.btn_zoom_reset.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.btn_selection.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
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
        energyList = greycoprops(glcm4D, 'energy')

        mu = moments_central(imageArray)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)

        contrastList = np.hstack(contrastMatrix)
        homogeneityList = np.hstack(homogeneityMatrix)
        energyList = np.hstack(energyList)
        hu = np.hstack(hu)
        # print(f"Hu moments: {hu}")


        return list(contrastList) + list(homogeneityList) + list(entropyList) + list(energyList) + list(hu)


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
        runQuantity = 100
        greatestAcuracy = 0;
        smallestAcuracy = 0;
        accuracyMedia = 0;
        specificityMedia = 0;
        greatestSpecificity = 0;
        smallestSpecificity = 0;
        mean_sensibilityMedia = 0;
        greatestSensibility = 0;
        smallestSensibility = 0;
        for i in range(0, runQuantity):
          print(f"Iteracao: {i}")
          X_train, X_test, y_train, y_test = train_test_split(imagesDescriptors, #varianceTrashHold(np.array(imagesDescriptors)),
                                                              types,
                                                              test_size=.25)

          # self.clf = svm.SVC(kernel='linear', probability=True, gamma="scale", C=1.0)
          clf_selected = make_pipeline(
                  SelectKBest(f_classif, k=2), MinMaxScaler(), LinearSVC()
          )
          print("Iniciando treinamento do SVM")
          clf_selected.fit(X_train, y_train)
          # self.clf.fit(X_train, y_train)
          infoString = ""
          y_predicted = clf_selected.predict(X_test)
          # y_predicted = self.clf.predict(X_test)
          # print(f"Predicted Values {y_predicted}")
          # print(f"Expected Values {y_test}")

          accuracy = accuracy_score(y_test, y_predicted)
          confusionMatrix = confusion_matrix(y_test, y_predicted)

          # print(confusionMatrix)
          infoString += str(confusionMatrix)
          (mean_sensibility, specificity) = self.computeMetrics(confusionMatrix)
          print(f"Accuracy {accuracy}")

          accuracyMedia += accuracy * 100;
          specificityMedia += specificity * 100;
          mean_sensibilityMedia += mean_sensibility * 100;
         
          if(i == 0):
            greatestAcuracy = accuracy
            smallestAcuracy = accuracy
            greatestSpecificity = specificity
            smallestSpecificity = specificity
            greatestSensibility = mean_sensibility
            smallestSensibility = mean_sensibility
          else:
            if(accuracy >= greatestAcuracy):
              greatestAcuracy = accuracy
            elif(accuracy <= smallestAcuracy):
              smallestAcuracy = accuracy

            if(specificity >= greatestSpecificity):
              greatestSpecificity = specificity
            elif(accuracy <= smallestSpecificity):
              smallestSpecificity = specificity

            if(mean_sensibility >= greatestSensibility):
              greatestSensibility = mean_sensibility
            elif(mean_sensibility <= smallestSensibility):
              smallestSensibility = mean_sensibility

          infoString += f"\nAccuracy {accuracy}"
          infoString += f"\nSensiblidade Média: {mean_sensibility}"
          infoString += f"\nEspecificidade: {specificity}"
          
        accuracyMedia = accuracyMedia / runQuantity
        specificityMedia = specificityMedia / runQuantity
        mean_sensibilityMedia = mean_sensibilityMedia / runQuantity

        print(f"Accurracy avarage {accuracyMedia}")
        print(f"Specificity avarage {specificityMedia}")
        print(f"Sensibility avarage {mean_sensibilityMedia}")

        print(f"Greatest Accuracy {greatestAcuracy}")
        print(f"Smallest Accuracy {smallestAcuracy}")

        print(f"Greatest Specificity {greatestSpecificity}")
        print(f"Smallest Specificity {smallestSpecificity}")

        print(f"Greatest Sensibility {greatestSensibility}")
        print(f"Smallest Sensibility {smallestSensibility}")

        # return self.clf
        return clf_selected

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
