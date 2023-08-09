
import numpy as np
import cv2
import sys, math
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as mpl
from skimage import io, img_as_float, img_as_ubyte
from filters import *


#Aviso: en cada sitio que busco los valores de los arrays son distintos,en caso de duda cambiar los valores
def robertsOP():
    return np.array([[-1,0],[0,1]]),np.array([[0,-1],[1,0]])

def centralOP():
    return np.array([[-1,0,1]]),np.array([[-1],[0],[1]])

def prewittOP():
    return np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

def sobelOP():
    return np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),np.array([[-1,-2,-1],[0,0,0],[1,2,1]])


def gradientImage(inImage, operator):
    if operator == 'Roberts':
        x, y = robertsOP()
    elif operator == 'CentralDiff':
        x, y = centralOP()
    elif operator == 'Sobel':
        x, y = sobelOP()
    elif operator == 'Prewitt':
        x, y = prewittOP()
    else:
        return print("Inexistente")

    gx = filterImage(inImage, x)
    gy = filterImage(inImage, y)

    return gx, gy


def edgeCanny(inImage, sigma, tlow, thigh):
    # Aplicar filtro gaussiano a la imagen de entrada
    blurredImage = gaussianFilter(inImage, sigma)

    # Calcular los gradientes en las direcciones x e y
    [gradientX, gradientY] = gradientImage(blurredImage, "Sobel")

    M = inImage.shape[0]
    N = inImage.shape[1]
    magnitude = np.zeros(inImage.shape)  # Matriz para almacenar la magnitud de los gradientes
    orientation = np.zeros(inImage.shape)  # Matriz para almacenar la orientación de los gradientes

    # Calcular la magnitud y orientación de los gradientes
    for row in range(M):
        for col in range(N):
            magnitude[row, col] = np.sqrt(pow(gradientX[row, col], 2) + pow(gradientY[row, col], 2))
            orientation[row, col] = np.arctan2(gradientY[row, col], gradientX[row, col])

    magnitude = magnitude / magnitude.max() * 255  # Normalizar la magnitud de los gradientes en el rango [0, 255]

    outputImage = np.zeros(inImage.shape)  # Matriz para almacenar la imagen de salida

    # Calcular los píxeles de borde
    for row in range(M):
        for col in range(N):
            try:
                q = 255
                r = 255

                # Determinar los píxeles vecinos en función de la orientación del gradiente
                if (0 <= orientation[row, col] < 22.5) or (157.5 <= orientation[row, col] <= 180):
                    q = magnitude[row, col + 1]
                    r = magnitude[row, col - 1]
                elif (22.5 <= orientation[row, col] < 67.5):
                    q = magnitude[row + 1, col - 1]
                    r = magnitude[row - 1, col + 1]
                elif (67.5 <= orientation[row, col] < 112.5):
                    q = magnitude[row + 1, col]
                    r = magnitude[row - 1, col]
                elif (112.5 <= orientation[row, col] < 157.5):
                    q = magnitude[row - 1, col - 1]
                    r = magnitude[row + 1, col + 1]

                # Comprobar si el píxel actual es máximo local en la dirección del gradiente
                if (magnitude[row, col] >= q) and (magnitude[row, col] >= r):
                    outputImage[row, col] = magnitude[row, col]  # Asignar el valor del píxel actual a la imagen de salida
                else:
                    outputImage[row, col] = 0  # El píxel no es máximo local, asignar cero

            except IndexError as e:
                pass

    # Umbralización con histéresis
    for row in range(M):
        for col in range(N):
            if (outputImage[row, col] <= tlow):  # Comprobar si el píxel es menor que el umbral inferior
                try:
                    # Comprobar si hay algún vecino que supere el umbral superior
                    if (
                        (outputImage[row + 1, col - 1] <= thigh)
                        or (outputImage[row + 1, col] <= thigh)
                        or (outputImage[row + 1, col + 1] <= thigh)
                        or (outputImage[row, col - 1] <= thigh)
                        or (outputImage[row, col + 1] <= thigh)
                        or (outputImage[row - 1, col - 1] <= thigh)
                        or (outputImage[row - 1, col] <= thigh)
                        or (outputImage[row - 1, col + 1] <= thigh)
                    ):
                        outputImage[row, col] = thigh
                    else:
                        outputImage[row, col] = 0

                except IndexError as e:
                    pass

    return 255 - outputImage








inImage=cv2.imread('image0.png')
inImage=cv2.cvtColor(inImage,cv2.COLOR_BGR2GRAY)
ope = "CentralDiff"
[outX, outY] = gradientImage(inImage, ope)

io.imsave(f"./resultados/res{ope}X.png", outX)
io.imsave(f"./resultados/res{ope}Y.png", outY)





# inImage=cv2.imread('circles1.png')
# inImage=cv2.cvtColor(inImage,cv2.COLOR_BGR2GRAY)
# inImage=np.flipud(inImage)
# low = 0.2
# high = 0.2

# outlows = edgeCanny(inImage, 0.4, low, low)
# io.imsave("./resultados/lows.png", outlows)

# outhighs = edgeCanny(inImage, 0.4, high, high)
# io.imsave("./resultados/Cannyhighs.png", outhighs)

# outMezcla = edgeCanny(inImage, 0.4, low, high)
# io.imsave("./resultados/CannyMezcla.png", outMezcla)


cv2.waitKey(0)