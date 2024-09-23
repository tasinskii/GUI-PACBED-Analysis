import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import random
from scipy import ndimage
import cv2

def alter_image(img):
    #noise
    PEAK = 1
    noise = np.random.poisson(img * 255.0 * PEAK) / PEAK / 255
    noisy = img + noise
    
    #random rotation
    rand_angle = random.randint(0,90)
    rotated = ndimage.rotate(noisy, rand_angle)
    final = rotated
    #random translation
    x, y = final.shape
    shift = random.randint(-10,10)
    rolled = np.roll(final, shift, axis=[0, 1])
    rolled = cv2.rectangle(rolled, (0, 0), (x, shift), 0, -1)
    rolled = cv2.rectangle(rolled, (0, 0), (shift, y), 0, -1)
    final = rolled
    #plt.imshow(final, cmap='inferno')
    #plt.show()
    
    return noisy
    
for i in range(0,150): 
    
    img = np.load("STO-Data/STO-unaugmented/pacbed-" + str(i) + "-STO.npy")
    print(img.shape)
    for j in range(0,20):
        res = alter_image(img)
        np.save("STO-Data/STO/"+str(j)+"_"+str(i),res)
