
import numpy as np
import cv2
import sys, math
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from histograms import *

def filterImage(inImage, kernel):

    M = inImage.shape[0]  # Obtener el número de filas de la imagen
    N = inImage.shape[1]  # Obtener el número de columnas de la imagen

    if kernel.ndim == 1:
        kernel_height = kernel.shape[0]  # Obtener la altura del kernel
        kernel_width = 1  # La anchura del kernel es 1 para un kernel unidimensional
    else:
        kernel_height = kernel.shape[0]  # Obtener la altura del kernel
        kernel_width = kernel.shape[1]  # Obtener la anchura del kernel

    outImage = np.zeros(inImage.shape)  # Crear una matriz de salida con las mismas dimensiones que la imagen de entrada

    height = (kernel_height - 1) // 2  # Calcular el desplazamiento vertical necesario para aplicar el kernel
    width = (kernel_width - 1) // 2  # Calcular el desplazamiento horizontal necesario para aplicar el kernel

    image = np.zeros((M + (2 * height), N + (2 * width)))  # Crear una imagen con relleno

    # Copiar la imagen original en la imagen con relleno
    image[height:image.shape[0] - height, width:image.shape[1] - width] = inImage

    for row in range(M):
        for col in range(N):
            # Realizar la convolución del kernel con la sección correspondiente de la imagen con relleno
            outImage[row, col] = np.sum(kernel * image[row:row + kernel_height, col:col + kernel_width])

    return outImage


def createDelta(dims: Tuple[int, int] = (256, 256)) -> np.ndarray :
    blk = np.zeros(dims)

    blk[int(dims[0]/2), int(dims[1]/2)] = 1

    return blk



def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma) + 1)  # Calcular el tamaño del kernel basado en el valor de sigma

    result = np.zeros(N)  # Aarray de ceros para almacenar los valores del kernel

    mid = int(N / 2)  # Calcular la posición central del kernel

    # fórmula de la función de distribución Gaussiana
    result = np.array([(1 / (np.sqrt(2 * np.pi) * sigma)) * (1 / (np.exp((i ** 2) / (2 * sigma ** 2)))) for i in range(-mid, mid + 1)])

    return result / sum(result)  # Normalizar el kernel dividiendo por la suma de sus valores






def gaussianFilter (inImage, sigma):

    gausskernel = gaussKernel1D(sigma)
    out = filterImage(inImage, gausskernel)
    out2 = filterImage(out, np.transpose(gausskernel))

    return out2




def medianFilter(inImage: np.ndarray, filterSize: int) -> np.ndarray:
    
    dim = np.shape(inImage)  # Dimensiones de la imagen de entrada
    outImage = np.empty(dim)  # Crear una matriz vacía para almacenar la imagen de salida
    mitad = int((filterSize / 2))  # Calcular la mitad del tamaño del filtro

    for row in range(0, dim[0]):  # Iterar por cada fila de la imagen
        for col in range(0, dim[1]):  # Iterar por cada columna de la imagen

            # Calcular los límites superior e inferior de la submatriz basados en la posición actual y el tamaño del filtro
            left = 0 if (row - mitad) < 0 else row - mitad
            right = dim[0] - 1 if (row + mitad) > (dim[0] - 1) else row + mitad + 1
            top = 0 if (col - mitad) < 0 else col - mitad
            bottom = dim[1] - 1 if (col + mitad) > (dim[1] - 1) else col + mitad + 1

            submatrix = inImage[left:right, top:bottom]  # Extraer la submatriz de la imagen original
            outImage[row, col] = np.median(submatrix)  # Calcular el valor mediano de la submatriz y asignarlo a la imagen de salida

    return outImage  # Devolver la imagen de salida filtrada




def highBoost(inImage,A,method,param):
	#condiciones
    if (method is None) or (param is None):
        return inImage

    if method == 'gaussian' :
        outImage = gaussianFilter(inImage,param) 
    else :
    	if method == 'median' :
    		#outImage = medianFilter(inImage,int(param))
        	outImage = medianFilter(inImage,int(param))
    	else:
        	return None

    if A >= 0:
        #Si es positivo guardamos informacion de la imagen
        inImage = np.multiply(inImage,A).astype(np.uint8)
        return np.subtract(inImage,outImage)
    else:
        return outImage



#filter_image example 
delta = createDelta()
io.imsave("./resultados/delta.png", delta)


# kernel=cv2.imread('kernel1.png')
# kernel=cv2.cvtColor(kernel,cv2.COLOR_BGR2GRAY)
# delta=cv2.imread('delta.png')
# delta=cv2.cvtColor(delta,cv2.COLOR_BGR2GRAY)

# # M = inImage.shape[0]  # Obtener el número de filas de la imagen
# # N = inImage.shape[1]  # Obtener el número de columnas de la imagen
# # print(M)
# # print(N)
# #delta=cv2.cvtColor(delta,cv2.COLOR_BGR2GRAY)
# # laplacian = np.array((
# #     [0, 1, 0],
# #     [1, -4, 1],
# #     [0, 1, 0]), dtype="int")

# outImage2=filterImage(delta,kernel[:-1,:-1])
# io.imsave("./resultados/pruebakernel.png", outImage2)
#cv2.imshow('original',gris)
#cv2.imshow('mi filtro',outImage2)



#######KERNEL FUNCTION####################
# salida=gaussKernel1D(10)
# print(salida)
# plt.plot(salida)

# plt.show()




#######GAUSSIAN FUNCTION####################
# gauss=cv2.imread('delta.png')
# gris=cv2.cvtColor(gauss,cv2.COLOR_BGR2GRAY)
# outImage1=gaussianFilter(gris, 5)
# #outImage1 = adjustIntensity(outImage1, outRange=[0, 255])
# #outImage2=cv2.GaussianBlur(gris,(7,7),0)
# # cv2.imshow('original',gris)
# # cv2.imshow('mi filtro',outImage1)
# imgMod = gaussianFilter(gris, 10)
# io.imsave("./resultados/gaussianFilter.png", adjustIntensity(imgMod, [], [0, 255]))
# plotgaussian=cv2.imread('gaussianFilter.png')
# #(hist, binsIn) = np.histogram(plotgaussian, 256)
# plt.plot(256, imgMod)

# plt.show()




##########MEDIAN FUNCTION####################
# lisa = cv2.imread('grid.png')
# img_median = cv2.medianBlur(lisa, 5)

# cv2.imshow("median", np.hstack((lisa, img_median)))
# img = Image.open("grid.png").convert("L")
# arr = np.array(img)
# removed_noise = medianFilter(arr, 5) 
# img = Image.fromarray(removed_noise)
# img.show()


#######HIGH FUNCTION####################
# img = Image.open("image2.png").convert("L")
# arr = np.array(img)
# outImage1=highBoost(arr,-3,"median",5)
# #En caso de gaussian poner lo de abajo
# #outImage1 = adjustIntensity(outImage1, outRange=[0, 255])
# #io.imsave("./resultados/highGaussianFilter.png", adjustIntensity(imgMod, [], [0, 255]))
# img = Image.fromarray(outImage1)
# img.show()





cv2.waitKey(0)