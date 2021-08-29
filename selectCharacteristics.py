import tkinter as tk
from PIL import ImageTk

class SelectCharacteristics:

    def showWindow(self, selectedCharacteristics, selectedCharacteristicsListCallBack):
        self.window = tk.Toplevel()
        self.window.title("Select Characteristics")
        self.window.geometry('250x100')

        self.selectedCharacteristics = selectedCharacteristics
        self.selectedCharacteristicsListCallBack = selectedCharacteristicsListCallBack

        self.setVars()
        self.setMainContainer()

    def setVars(self):
        self.homogeneity = tk.IntVar()
        self.contrast = tk.IntVar()
        self.energy = tk.IntVar()
        self.entropy = tk.IntVar()
        self.hu = tk.IntVar()

        if ('Homogeneity' in self.selectedCharacteristics):
            self.homogeneity.set(1)
        else:
            self.homogeneity.set(0)
        if ('Contrast' in self.selectedCharacteristics):
            self.contrast.set(1)
        else:
            self.contrast.set(0)
        if ('Energy' in self.selectedCharacteristics):
            self.energy.set(1)
        else:
            self.energy.set(0)
        if ('Entropy' in self.selectedCharacteristics):
            self.entropy.set(1)
        else:
            self.entropy.set(0)
        if ('Hu' in self.selectedCharacteristics):
            self.hu.set(1)
        else:
            self.hu.set(0)

    def setMainContainer(self):
        tk.Checkbutton( self.window, text="Homogeneity", variable=self.homogeneity, command=self.updateList).grid(row=0, column=0, sticky=tk.W)

        tk.Checkbutton( self.window, text="Entropy", variable=self.entropy, command=self.updateList).grid(row=0, column=1, sticky=tk.W)

        tk.Checkbutton( self.window, text="Energy", variable=self.energy, command=self.updateList).grid(row=1, column=0, sticky=tk.W)

        tk.Checkbutton( self.window, text="Contrast", variable=self.contrast, command=self.updateList).grid(row=1, column=1, sticky=tk.W)

        tk.Checkbutton( self.window, text="Hu moment invariants", variable=self.hu,  command=self.updateList).grid(row=2, column=0, sticky=tk.W)

    def updateList(self):
        selected_list = []
        if (self.homogeneity.get() == 1):
            selected_list.append('homogeneity')

        if (self.contrast.get() == 1):
            selected_list.append('contrast')

        if (self.energy.get() == 1):
            selected_list.append('energy')

        if (self.entropy.get() == 1):
            selected_list.append('entropy')

        if (self.hu.get() == 1):
            selected_list.append('hu')

        self.selectedCharacteristicsListCallBack(selected_list)
