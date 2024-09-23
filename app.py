# pacbed analysis tool main app 
# author: Timofei Asinski
# Summer 2024

import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QGridLayout, QWidget, QLineEdit, QPushButton
from PyQt6.QtGui import QPixmap, QColor, QImage, QPainter, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import csv
import numpy as np
import math
import py4DSTEM as p4d
import cv2
import threading
import mahotas
import matplotlib as mpl
from keras.models import load_model

#init gui window
class init_gui_window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mw = None
        self.setMinimumSize(200,200)
        self.setWindowTitle("PACBED Analysis GUI Initialization")
        self.layout = QGridLayout()
        
        ### initialization section: h5 and bf
        bf_label = QLabel("Input filepath to brightfield image: ")
        bf_filepath_line = QLineEdit()
        self.layout.addWidget(bf_label,0,0)
        self.layout.addWidget(bf_filepath_line,0,1)
        h5_label = QLabel("Input filepath to 4D STEM .h5")
        scanshape_label = QLabel("4D STEM scan shape (optional): ")
        h5_filepath_line = QLineEdit()
        scanshape_line_x = QLineEdit(text="x")
        scanshape_line_y = QLineEdit(text="y")
        load_data_button = QPushButton(text="Load data")
        self.layout.addWidget(h5_label, 1, 0)
        self.layout.addWidget(h5_filepath_line, 1, 1)
        self.layout.addWidget(scanshape_label, 2, 0)
        self.layout.addWidget(scanshape_line_x, 2, 1)
        self.layout.addWidget(scanshape_line_y, 2, 2)
        load_data_button.clicked.connect(lambda: self.initialize_data(bf_filepath_line, h5_filepath_line, scanshape_line_x, scanshape_line_y))
        self.layout.addWidget(load_data_button,3,0)

        #set layout
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def initialize_data(self, line_bf, line_h5, scan_x, scan_y):
        self.mw = main_window(line_h5.text(), line_bf.text(), scan_x.text(), scan_y.text())
        self.mw.show()



class main_window(QWidget):
    def __init__(self, h5_path, bf_path, scan_x, scan_y):
        super().__init__()
        self.layout = QGridLayout()
        self.h5path = h5_path
        self.h5_status = "(Currently loading datacube, please wait before displaying diffraction patterns)"
        self.bfpath = bf_path
        self.points = []
        self.scanshape = (int(scan_x), int(scan_y))
        self.dc = None
        self.thickness_range = None
        self.model = None
        self.current_dp = None
        
        
        self.h5_label = QLabel("Reading 4D STEM data from: " + self.h5_status) 

        top_label = QLabel("Brightfield Image: ")
        top_label2 = QLabel("Diffraction Pattern: ")
        self.bflabel = QLabel()
        self.dplabel = QLabel()
        bf_img = QPixmap(self.bfpath).scaled(512,512) #TODO: func to figure out how to resize and maintain point coherency
        dp_img = QPixmap(512, 512)
        dp_img.fill(QColor('darkGray'))
        self.bflabel.setPixmap(bf_img)
        self.dplabel.setPixmap(dp_img) #placeholder
        self.dp_line = QLineEdit(text="dp.jpg")
        clear_button = QPushButton(text="Clear selected points")
        dp_line_label = QLabel("Save diffraction pattern image as: ")
        gen_dp_button = QPushButton(text="Derive diffraction pattern")
        self.model_label = QLabel("No model currently loaded")
        self.thickness_label = QLabel("Thickness estimate: ")


        
        thickness_button = QPushButton(text="Determine region thickness")
        

        self.model_line = QLineEdit(text="Enter path to model .h5 here")
        load_cnn_button = QPushButton(text="Load model")
        
        self.layout.addWidget(self.h5_label, 0, 0)
        self.layout.addWidget(top_label, 1, 0)
        self.layout.addWidget(top_label2, 1, 1)
        self.layout.addWidget(self.bflabel, 2, 0)
        self.layout.addWidget(clear_button, 3, 0)
        self.layout.addWidget(gen_dp_button, 0,1)
        self.layout.addWidget(thickness_button, 3,2)
        self.layout.addWidget(self.dplabel, 2, 1)
        self.layout.addWidget(self.model_line, 4, 0)
        self.layout.addWidget(load_cnn_button, 5, 0)
        self.layout.addWidget(self.dp_line, 1, 2)
        self.layout.addWidget(dp_line_label, 0, 2)
        self.layout.addWidget(self.model_label, 3, 1)
        self.layout.addWidget(self.thickness_label, 2, 2)

        clear_button.clicked.connect(self.clear_bf)
        gen_dp_button.clicked.connect(self.gen_dp)
        load_cnn_button.clicked.connect(self.load_cnn)
        thickness_button.clicked.connect(self.analyze_dp)

        self.setLayout(self.layout)
        self.t = threading.Thread(target=self.instantiate_dc)
        self.t.start() # TODO add spinny thing for loading
        
    def convert_pt(self, point): #point (x,y)
        bf = cv2.imread(self.bfpath)
        converted_x = int((point[0]/512) * bf.shape[0])
        converted_y = int((point[1]/512) * bf.shape[1])
        return (converted_x, converted_y)

    def instantiate_dc(self):
        self.dc = p4d.import_file(self.h5path)
        self.dc.set_scan_shape((self.scanshape[0], self.scanshape[1]))
        self.h5_status = self.h5path
        self.h5_label.setText("Reading 4D STEM data from: " + self.h5_status) 

    def gen_dp(self): #TODO add option to name dp file
        path = self.dp_line.text()
        poly = self.points
        print(poly)
        print(self.points)
        canvas = np.zeros(self.scanshape)
        mahotas.polygon.fill_polygon(poly, canvas)
        pts = np.argwhere(canvas)
        dps = []
        for i in pts:
            dps.append(self.dc[i[0], i[1], :, :])


        dp_avg = np.mean(dps, axis=0)
        mpl.image.imsave(path, dp_avg, cmap='gray', vmax=9)
        self.dplabel.setPixmap(QPixmap("dp.jpg").scaled(512,512))
        self.current_dp = path

    def mouseDoubleClickEvent(self, e):
        canvas = self.bflabel.pixmap()
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(204,0,0)) 
        painter = QPainter(canvas)
        painter.setPen(pen)
        painter.drawEllipse(int(e.position().x())-22, int(e.position().y())-90, 5, 5)
        point = (int(e.position().x()) - 22, int(e.position().y())-90) #TODO Fix points
        self.points.append(self.convert_pt(point))
        painter.end()
        self.bflabel.setPixmap(canvas)

    def clear_bf(self):
        self.points = []
        bf_img = QPixmap(self.bfpath).scaled(512,512)
        self.bflabel.setPixmap(bf_img)

    def load_cnn(self):
        model_path = self.model_line.text()
        self.model = load_model(model_path) # load from path
        self.model_label.setText("Loaded from: " + model_path)

    def analyze_dp(self):
        im = cv2.imread(self.current_dp)
        expanded_im = np.expand_dims(im, axis=0)
        result = np.argwhere(self.model(expanded_im).numpy())[0][1]
        self.thickness_label.setText("sample thickness: " + str(result) + " nm") 


app = QApplication([])
w = init_gui_window()
w.show()
app.exec()
