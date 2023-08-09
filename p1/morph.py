
import numpy as np
import cv2
import sys, math
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as mpl
from skimage import io, img_as_float


def Choose(inImage: np.ndarray, SE: np.ndarray, center: List, type: str) -> np.ndarray:
    # Verificar el tipo de operación morfológica y asignar valores correspondientes
    if type == "erode":
        color = float(255)
        sel = np.min
    elif type == "dilate":
        color = float(0)
        sel = np.max
    else:
        raise Exception("type not valid")  # Lanzar una excepción si el tipo no es válido

    tamImage = np.shape(inImage)  # Obtener dimensiones de la imagen de entrada
    tamSE = np.shape(SE)  # Obtener dimensiones del elemento estructurante

    if len(center) < 2:
        center = [tamSE[0] // 2 + 1, tamSE[1] // 2 + 1]  # Calcular el centro del elemento estructurante si no se proporciona

    outImage = np.empty(tamImage)  # Crear una matriz vacía para la imagen de salida

    # Iterar sobre los píxeles de la imagen de entrada
    for row in range(tamImage[0]):
        for col in range(tamImage[1]):
            # Calcular los límites de la sección de la imagen y el elemento estructurante a considerar
            limitLeft = max(row - (center[0] - 1), 0)
            limitRight = min(row + (tamSE[0] - center[0]), tamImage[0] - 1)
            limitTop = max(col - (center[1] - 1), 0)
            limitBottom = min(col + (tamSE[1] - center[1]), tamImage[1] - 1)

            submatrix = inImage[limitLeft:limitRight+1, limitTop:limitBottom+1]  # Extraer la submatriz correspondiente
            dimsSub = np.shape(submatrix)  # Obtener dimensiones de la submatriz

            template = np.full(tamSE, color)  # Crear una plantilla con valores iniciales
            leftTemplate = max(-limitLeft, 0)
            topTemplate = max(-limitTop, 0)

            # Asignar los valores de la submatriz a la plantilla en la posición correcta
            template[leftTemplate:leftTemplate+dimsSub[0], topTemplate:topTemplate+dimsSub[1]] = submatrix

            # Aplicar la operación morfológica y asignar el resultado a la imagen de salida
            outImage[row, col] = sel(template * SE)

    return outImage



#Operadores Morfologicos

def erode(inImage: np.ndarray, SE: np.ndarray, center: List = []) -> np.ndarray :
	return Choose(inImage, SE, center, "erode")


def dilate(inImage: np.ndarray, SE: np.ndarray, center: List = []) -> np.ndarray :
	return Choose(inImage, SE, center, "dilate")


def opening(inImage: np.ndarray, SE: np.ndarray, center: List = []) -> np.ndarray :
	return dilate((erode(inImage, SE, center)), SE, center)


def closing(inImage: np.ndarray, SE: np.ndarray, center: List = []) -> np.ndarray :
	return erode((dilate(inImage, SE, center)), SE, center)


def fill(inImage: np.ndarray, seeds: np.ndarray, SE: np.ndarray = [], center: list = []) -> np.ndarray:
    # Verificar si no se proporcionó un elemento estructurante y asignar uno por defecto
    if len(SE) == 0:
        SE = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    tam_SE = np.shape(SE)  # Obtener dimensiones del elemento estructurante

    if len(center) < 2:
        center = [math.floor(tam_SE[0] / 2) + 1, math.floor(tam_SE[1] / 2) + 1]  # Calcular el centro del elemento estructurante si no se proporciona

    condition = True
    while condition:
        group = []

        for seed in seeds:
            limitLeft = seed[0] - (center[0] - math.floor(tam_SE[0] / 2))
            limitTop = seed[1] - (center[1] - math.floor(tam_SE[1] / 2))

            for i in range(tam_SE[0]):
                for j in range(tam_SE[1]):
                    imgI = limitLeft + i
                    imgJ = limitTop + j

                    if (np.all(SE[i, j] == 1)) and (np.all(0 <= imgI) and np.all(imgI < inImage.shape[0])) and (np.all(0 <= imgJ) and np.all(imgJ < inImage.shape[1])) and (np.all(inImage[imgI, imgJ] == 0)):
                        inImage[imgI, imgJ] = 255
                        if (i + 1 != center[0] or j + 1 != center[1]):
                            if len(group) == 0:
                                group = [[imgI, imgJ]]
                            else:
                                group = np.append(group, [[imgI, imgJ]], axis=0)

        condition = (len(group) != 0)  # Verificar si hay nuevas semillas para continuar el proceso
        seeds = group

    return inImage




se =  [
	[1, 1, 1],
	[1, 1, 1],
	[1, 1, 1]
]

# dims = np.shape(inImage)
# centerImg = [math.floor(dims[0] / 2) + 1, math.floor(dims[1] / 2) + 1]

# inImage=cv2.imread('morph.png')
# inImage=cv2.cvtColor(inImage,cv2.COLOR_BGR2GRAY)
# outImage=erode(inImage,se)
# io.imsave("./resultados/probaErode.png", outImage)
# outImage=dilate(inImage,se)
# io.imsave("./resultados/probaDilate.png", outImage)
# outImage=closing(inImage,se)
# io.imsave("./resultados/probaClose.png", outImage)
# outImage=opening(inImage,se)
# io.imsave("./resultados/probaOpen.png", outImage)



# inImage=cv2.imread('image0.png')

# se2 = np.asarray([
# 		[1, 1, 1], 
# 		[1, 1, 1],
# 		[1, 1, 1]])
	

# dims = np.shape(inImage)
# centerImg = [math.floor(dims[0] / 2) + 1, math.floor(dims[1] / 2) + 1]

# print(centerImg)

# imFilled = fill(inImage, np.asarray([centerImg]), SE = se2)

# io.imsave("./resultados/Fill.png", imFilled)

cv2.waitKey(0)