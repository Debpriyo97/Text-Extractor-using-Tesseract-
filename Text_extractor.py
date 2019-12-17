import tkinter as tk
from tkinter import *
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import cv2
import sys
import pytesseract
from PIL import Image, ImageOps
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
from PIL import Image
import pytesseract
import glob

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Vgg16 utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess 

# Resnet utils 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# utils to dump 
from pickle import dump
# import glob
import glob

from PIL import Image, ImageOps
import os
import cv2
import sys
import pytesseract
from PIL import Image, ImageOps
import tempfile
import numpy as np
import matplotlib.pyplot as plt



global fd
fd= pd.DataFrame()


LARGE_FONT= ("Verdana", 12)



class TextExtractor(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)
        container.configure(background='black')
    
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


        
       
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)

        tk.Frame.configure(self,background='black')  
        label = tk.Label(self, text="Text Extractor", bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
        label.pack(pady=10,padx=10)
        label1 = tk.Label(self, text="Operations Available", bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 25, 'italic bold underline'))
        label1.pack(pady=30,padx=10)



        button2 = tk.Button(self, text="Text Extractor",fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '),
                            command=lambda: controller.show_frame(PageTwo))
        button2.place(x=800,y=300)

        button4 = tk.Button(self, text="EXIT",fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '), command= self.quit
                     )
        button4.place(x=1200,y=700)


        
class PageOne(tk.Frame):

   def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,background='black')       
        label = tk.Label(self, text="Document Classifier",bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
        label.pack(pady=10,padx=10)
        message10 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=50  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message10.place(x=700, y=600)
        message11 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=50  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message11.place(x=700, y=800)
        message12 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=50  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message12.place(x=700, y=400)



class PageTwo(tk.Frame):

   def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,background='black')       
        label = tk.Label(self, text="Text Extractor",bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
        label.pack(pady=10,padx=10)
        label = tk.Label(self, text="Multiple Images",bg="black"  ,fg="yellow"  ,width=30  ,height=3,font=('times', 20, 'italic bold underline'))
        label.place(x=550,y=150)
        label = tk.Label(self, text="Single Images",bg="black"  ,fg="yellow"  ,width=30  ,height=3,font=('times', 20, 'italic bold underline'))
        label.place(x=950,y=150)
        message = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=35  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message.place(x=500, y=400)
        message1 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=30  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message1.place(x=700, y=600)
        message2 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=55  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message2.place(x=1000, y=400)
        message3 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=30  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message3.place(x=1000, y=600)

        # Include tesseract executable in your path
        pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"



        def directory():
            global a
            directory = filedialog.askdirectory()
            a= directory
            message.configure(text= a)
        
        def file_select():
            global b
            path= filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            b= path
            message2.configure(text=b)
    

        
        
        IMAGE_SIZE = 1800
        BINARY_THREHOLD = 180
        



        def process_image_for_ocr(file_path):
            
            temp_filename = set_image_dpi(file_path)
            #im_new=remove_noise_and_smooth(temp_filename)
            return (temp_filename)

        def set_image_dpi(file_path):
            im = Image.open(file_path)
            im = ImageOps.expand(im, border=15)
            length_x, width_y = im.size
            factor = max(1, int(IMAGE_SIZE / length_x))
            size = factor * length_x, factor * width_y
            im_resized = im.resize(size, Image.ANTIALIAS)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_filename = temp_file.name
            im_resized.save(temp_filename, dpi=(300, 300))
            return temp_filename

        def image_smoothening(img):
            ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
            ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blur = cv2.GaussianBlur(th2, (1, 1), 0)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th3

        def remove_noise_and_smooth(file_name):
            img = plt.imread(file_name)
            filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,3)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            img = image_smoothening(img)
            or_image = cv2.bitwise_or(img, closing)
            plt.imshow(or_image)
            return or_image

        def text_extract():
            f=open("C:\\AAI\\Project_NDA\\output-1.txt", "a+")
            f.truncate(0)
            f.close()
            directory=a
            count=1
            for file in os.listdir(directory):
                
                filename = os.fsdecode(file)
                if filename.endswith(".jpg"):
                    image = os.path.join(directory, filename)
                    config=()
                    text = pytesseract.image_to_string(process_image_for_ocr(image))
                    f=open("C:\\AAI\\Project_NDA\\output-1.txt", "a+")
                    f.write("\n\n\n Document %d \n\n" %count)
                    f.write(text)
                    count=count+1    
                else:
                    continue
            message1.configure(text="Extraction completed")

        def text_file_extract():

                    print(b[0:37])
                    f=open("C:\\AAI\\Project_NDA\\output-4.txt", "a+")
                    f.truncate(0)
                    f.close()
                    text = pytesseract.image_to_string(process_image_for_ocr(b))
                    f=open("C:\\AAI\\Project_NDA\\output-4.txt", "a+")
                    f.write(text)
                    message3.configure(text="Extraction completed")
        def open_output():

            os.startfile("C:\\AAI\\Project_NDA\\output-4.txt")
            message3.configure(text="")

        def open_output1():

            os.startfile("C:\\AAI\\Project_NDA\\output-1.txt")
            message1.configure(text="")
       
       


       
               

       
       
       
        text = tk.Button(self, text="Text Extractor", command= text_extract  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        text.place(x=700, y=500)

        back = tk.Button(self, text="Back", command=lambda: controller.show_frame(StartPage)  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        back.place(x=1600, y=800)

        dire = tk.Button(self, text="Select Directory", command= directory  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        dire.place(x=700, y=300)
        fil = tk.Button(self, text="Select File", command= file_select  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        fil.place(x=1100, y=300)
        ext = tk.Button(self, text="Text Extract ", command= text_file_extract  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        ext.place(x=1100, y=500)
        out = tk.Button(self, text="View Text", command= open_output  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        out.place(x=1100, y=700)
        out1 = tk.Button(self, text="View Text ", command= open_output1  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        out1.place(x=700, y=700)
        
 
        






app = TextExtractor()
app.mainloop()