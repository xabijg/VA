import numpy as np
import cv2
import sys, math
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as mpl
from skimage import io, img_as_float
from skimage import exposure
import os


imin = 0.0
imax = 255.0

#Alteracion rango dinamico
def adjustIntensity(inImage,inRange: List = [],outRange: List = [0, 1]) -> np.ndarray :

	if (len(inRange)!=0):
		imin=inRange[0]
		imax=inRange[1]
	else :
		imin=inImage.min()
		imax=inImage.max()

	#Formula transparencias
	outImage=outRange[0]+(((outRange[1]-outRange[0])*(inImage-imin))/(imax-imin))

	return outImage


def equalizeIntensity(inImage, nBins=256):
    # Calcula el histograma y los centros de los bins
    hist, bin_centers = exposure.histogram(inImage, nBins)

    # Calcula la función de distribución acumulada (CDF)
    cdf = hist.cumsum()  # Función de distribución acumulada
    cdf = cdf / float(cdf[-1])  # Normalización

    # Interpola linealmente los valores de entrada utilizando los centros de los bins y la CDF
    out = np.interp(inImage.flat, bin_centers, cdf)

    # Devuelve la salida reconfigurada a la forma original de la imagen de entrada
    return out.reshape(inImage.shape)




#Testeo



inImage=cv2.imread('circles1.png')


########################################################


# inImage_grey = cv2.imread('circles1.png',cv2.IMREAD_GRAYSCALE)

# outImage1 = adjustIntensity(inImage_grey, outRange=[0, 255])
# #outImage1=adjustIntensity(inImage,[], [0.3, 0.7])
# cv2.imshow('original',inImage)
# cv2.imshow('outImage1',outImage1)
# #mpl.hist(inImage.ravel(),256,[0,256]); mpl.show()	
# (hist, binsIn) = np.histogram(inImage, 256)
# (histOut, binsOut) = np.histogram(outImage1, 256)
# mpl.subplot(1, 2, 1)
# mpl.plot(binsIn[1:], hist)
# mpl.subplot(1, 2, 2)
# mpl.plot(binsOut[1:], histOut)
# mpl.show()


########################################################
inImage = cv2.imread('eq0.png',cv2.IMREAD_GRAYSCALE)
outImage2 = equalizeIntensity(inImage)
io.imsave("./resultados/Equalize.png", outImage2)

(hist, _) = np.histogram(inImage, 256)
(histOut, _) = np.histogram(outImage2, 256)
mpl.subplot(1, 2, 1)
mpl.plot(hist)
mpl.subplot(1, 2, 2)
mpl.plot(histOut)
mpl.show()


cv2.waitKey(0)