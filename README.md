# GUI-PACBED-Analysis
This is a simple GUI tool based on the paper (paper) that used a convolutional neural network to determine the thickness of Position Averaged CBED patterns (PACBED). This repository contains the GUI tool and the necessary scripts for model training, data simulation, and data augmentation. <br />
#Initialization: <br />
<img width="502" alt="Screenshot 2024-09-23 at 3 41 46 PM" src="https://github.com/user-attachments/assets/8ed3621f-c6b2-4729-8faf-d6cb2d76f796"><br />
#Main window: <br />
<img width="1431" alt="Screenshot 2024-09-23 at 3 44 35 PM" src="https://github.com/user-attachments/assets/c0548d9c-14c2-43bb-bbff-4902f4f236a0"><br />
Simply double click on the brightfield image to determine the endpoints of the desired scan area, and derive a diffraction pattern corresponding to it. After training a model on simulated data, you can use the model to to make a prediction on the thickness of the area based on the diffraction pattern. 
