import os
import cv2
import pydicom
from matplotlib import pyplot as plt
import numpy as np
for file in os.listdir('.'):
    if '.dcm' in file:
        dcm = pydicom.dcmread(file).pixel_array
        dcm.setflags(write=1)  # can overwrite

        dcm[dcm < 1200] = 0  # collapse black pixels into 0

        # bring in range (0,255). Assumes max value is 4096
        dcm = dcm / 4096
        dcm = dcm * 255
        dcm = np.floor(dcm).astype(np.uint8, copy=False)

        #plt.imshow(dcm, cmap='gray')
        #plt.show()

        cv2.imwrite(file[:-4] + '.png', dcm)
