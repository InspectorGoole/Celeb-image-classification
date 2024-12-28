import numpy as np
import pywt
import cv2

# insert image, it will  transoform the image using pywt from wavelet. Then it will return a new image 
def w2d(img, mode="haar", level=1): 
    imArray = img
    #Datatype conversion
    #convert to gray scale
    imArray = cv2.cvtColor( imArray, cv2.COLOR_RGB2GRAY )
    # convert to float
    imArray = np.float32( imArray )
    imArray /= 255;
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    #Process_Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    #reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    return imArray_H

# that is the wavelet transform which is crucial for the computers becasue it differentiates between different fucntions
# it becomes difficult for a classifer to idenitify an image that has multiple colors.
# this helps computer differentiate between images, cuz it goes to the details of the shape and size of the features of the face
