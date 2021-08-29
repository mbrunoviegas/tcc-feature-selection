import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join
import SimpleITK as sitk
import math
from skimage.feature import greycomatrix, greycoprops 
from skimage.measure import moments_hu, moments_central, moments_normalized
from datetime import datetime
import numpy as np


class Characteristics():
    
    def showWindow(self, props):
        self.window_characteristics = tk.Tk()
        self.window_characteristics.title("Características")

        Label(self.window_characteristics, text="Homogeneity:", width=30).grid(row=0, column=0)
        Label(self.window_characteristics, text="  C1: "+ str(props[0])).grid(row=1, column=0)
        Label(self.window_characteristics, text="  C2: "+ str(props[1])).grid(row=2, column=0)
        Label(self.window_characteristics, text="  C4: "+ str(props[2])).grid(row=3, column=0)
        Label(self.window_characteristics, text="  C8: "+ str(props[3])).grid(row=4, column=0)
        Label(self.window_characteristics, text=" C16: "+ str(props[4])).grid(row=5, column=0)

        Label(self.window_characteristics, text="Entropy:", width=30).grid(row=0, column=1)
        Label(self.window_characteristics, text="  C1: "+str(props[15])).grid(row=1, column=1)
        Label(self.window_characteristics, text="  C2: "+str(props[16])).grid(row=2, column=1)
        Label(self.window_characteristics, text="  C4: "+str(props[17])).grid(row=3, column=1)
        Label(self.window_characteristics, text="  C8: "+str(props[18])).grid(row=4, column=1)
        Label(self.window_characteristics, text=" C16: "+str(props[19])).grid(row=5, column=1)

        Label(self.window_characteristics, text="", width=30).grid(row=6, column=0)

        Label(self.window_characteristics, text="Energy:", width=30).grid(row=7, column=0)
        Label(self.window_characteristics, text="  C1: "+str(0 + props[5])).grid(row=8, column=0)
        Label(self.window_characteristics, text="  C2: "+str(0 + props[6])).grid(row=9, column=0)
        Label(self.window_characteristics, text="  C4: "+str(0 + props[7])).grid(row=10, column=0)
        Label(self.window_characteristics, text="  C8: "+str(0 + props[8])).grid(row=11, column=0)
        Label(self.window_characteristics, text=" C16: "+str(0 + props[9])).grid(row=12, column=0)

        Label(self.window_characteristics, text="Contrast:", width=30).grid(row=7, column=1)
        Label(self.window_characteristics, text="  C1: "+str(0 + props[10])).grid(row=8, column=1)
        Label(self.window_characteristics, text="  C2: "+str(0 + props[11])).grid(row=9, column=1)
        Label(self.window_characteristics, text="  C4: "+str(0 + props[12])).grid(row=10, column=1)
        Label(self.window_characteristics, text="  C8: "+str(0 + props[13])).grid(row=11, column=1)
        Label(self.window_characteristics, text=" C16: "+str(0 + props[14])).grid(row=12, column=1)

        Label(self.window_characteristics, text="", width=30).grid(row=13, column=0)

        Label(self.window_characteristics, text="Hu moment invariants:", width=30).grid(row=14, column=0)
        Label(self.window_characteristics, text="  Moment 1: "+str(props[20])).grid(row=15, column=0)
        Label(self.window_characteristics, text="  Moment 2: "+str(props[21])).grid(row=16, column=0)
        Label(self.window_characteristics, text="  Moment 3: "+str(props[22])).grid(row=17, column=0)
        Label(self.window_characteristics, text="  Moment 4: "+str(props[23])).grid(row=18, column=0)
        Label(self.window_characteristics, text="  Moment 5: "+str(props[24])).grid(row=19, column=0)
        Label(self.window_characteristics, text="  Moment 6: "+str(props[25])).grid(row=20, column=0)
        Label(self.window_characteristics, text="  Moment 7: "+str(props[26])).grid(row=21, column=0)

        Label(self.window_characteristics, text="Tempo de execução:", width=30).grid(row=14, column=1)
        Label(self.window_characteristics, text=str(self.fim_characteristics - self.inicio_characteristics), width=30).grid(row=15, column=1)

        Label(self.window_characteristics, text="", width=30).grid(row=22, column=0)

        self.window_characteristics.mainloop()
        

    def CalcShowCharacteristics (self, image_array_gray, inicio_characteristics):
        characteristics_array = self.calcCharacteristics(image_array_gray, ["homogeneity","contrast","energy","entropy","hu"])

        self.inicio_characteristics = inicio_characteristics
        self.fim_characteristics = datetime.now()

        self.showWindow(characteristics_array)


    def calcCharacteristics (self, image_array_gray, selectedCharacteristics):
        self.matrix1 = []
        self.matrix2 = []
        self.matrix4 = []
        self.matrix8 = []
        self.matrix16 = []
        
        self.matrix1 = self.initMatrix(1, image_array_gray)
        self.matrix2 = self.initMatrix(2, image_array_gray)
        self.matrix4 = self.initMatrix(4, image_array_gray)
        self.matrix8 = self.initMatrix(8, image_array_gray)
        self.matrix16 = self.initMatrix(16, image_array_gray)
        return self.calcDescriptors(selectedCharacteristics, image_array_gray)
       

    
    def initMatrix(self,numMatrix, image_array_gray):
        matrix = []
        #criar matrixes de coocorrência
        matrix = greycomatrix(image_array_gray,[numMatrix],[0],32)
        matrix += greycomatrix(image_array_gray,[numMatrix],[math.pi/4],32)
        matrix += greycomatrix(image_array_gray,[numMatrix],[math.pi/2],32)
        matrix += greycomatrix(image_array_gray,[numMatrix],[(3*math.pi)/4],32)
        matrix += greycomatrix(image_array_gray,[numMatrix],[math.pi],32)
        matrix += greycomatrix(image_array_gray,[numMatrix],[(5*math.pi)/4],32)
        matrix += greycomatrix(image_array_gray,[numMatrix],[(3*math.pi)/2],32)
        matrix += greycomatrix(image_array_gray,[numMatrix],[(7*math.pi)/4],32)
        return matrix
      

    def calcDescriptors(self, selectedCharacteristics, image_array_gray):
        #calculo de hu
        mu = moments_central(image_array_gray)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)

        props = []

        if ('homogeneity' not in selectedCharacteristics and 
            'contrast' not in selectedCharacteristics and
            'energy' not in selectedCharacteristics and
            'entropy' not in selectedCharacteristics and
            'hu' not in selectedCharacteristics):
                selectedCharacteristics = ['homogeneity','contrast','energy','entropy','hu']

        if ('homogeneity' in selectedCharacteristics):
            props.append(greycoprops(self.matrix1, 'homogeneity')[0][0])
            props.append(greycoprops(self.matrix2, 'homogeneity')[0][0])
            props.append(greycoprops(self.matrix4, 'homogeneity')[0][0])
            props.append(greycoprops(self.matrix8, 'homogeneity')[0][0])
            props.append(greycoprops(self.matrix16, 'homogeneity')[0][0])

        if ('contrast' in selectedCharacteristics):
            props.append(greycoprops(self.matrix1, 'contrast')[0][0])
            props.append(greycoprops(self.matrix2, 'contrast')[0][0])
            props.append(greycoprops(self.matrix4, 'contrast')[0][0])
            props.append(greycoprops(self.matrix8, 'contrast')[0][0])
            props.append(greycoprops(self.matrix16, 'contrast')[0][0])

        if ('energy' in selectedCharacteristics):
            props.append(greycoprops(self.matrix1, 'energy')[0][0])
            props.append(greycoprops(self.matrix2, 'energy')[0][0])
            props.append(greycoprops(self.matrix4, 'energy')[0][0])
            props.append(greycoprops(self.matrix8, 'energy')[0][0])
            props.append(greycoprops(self.matrix16, 'energy')[0][0])

        if ('entropy' in selectedCharacteristics):
            props.append(math.sqrt(greycoprops(self.matrix1, 'energy')))
            props.append(math.sqrt(greycoprops(self.matrix2, 'energy')))
            props.append(math.sqrt(greycoprops(self.matrix4, 'energy')))
            props.append(math.sqrt(greycoprops(self.matrix8, 'energy')))
            props.append(math.sqrt(greycoprops(self.matrix16, 'energy')))

        if ('hu' in selectedCharacteristics):
            props.append(hu[0])
            props.append(hu[1])
            props.append(hu[2])
            props.append(hu[3])
            props.append(hu[4])
            props.append(hu[5])
            props.append(hu[6])

        return props
