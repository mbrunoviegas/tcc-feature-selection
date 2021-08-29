import tkinter as tk
from tkinter import *
from PIL import ImageTk

class Results:

    def show_results(self, result_string):       
        self.window = tk.Tk()
        self.window.title("Results")

        Label(self.window, text=(f'{result_string}')).grid(row=0, column=0)