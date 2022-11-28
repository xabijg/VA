import cv2
import numpy as np
from PIL import Image
from pylab import *
from skimage.exposure import rescale_intensity


#Variables

omin=0
omax=1
nbins=256
inRange=[]
outRange=[]

def compara_histogramas(inImage,inImage_mod):

    plt.hist(inImage_mod.ravel(), 256, [0, 1], color='r')
    plt.hist(inImage.ravel(), 256, [0, 1], color='b')#este no se muestra pero esta funcionando
    plt.legend(('original', 'modified'), loc='upper left')
    plt.show()



#adjustIntensity
def adjustIntensity(inImage,inRange,outRange):

	if(len(inRange)==0):inRange=[inImage.min(),inImage.max()]
	if(len(outRange)==0):outRange=[0,1]

	imin=inRange[0]
	imax=inRange[1]
	omin=outRange[0]
	omax=outRange[1]

	#Formula transparencias
	gnorm=omin+(((omax-omin)*(inImage-imin))/(imax-imin))
	#limites
	gnorm[gnorm>omax]=omax
	gnorm[gnorm<omin]=omin

	return gnorm


#equalizedHistogram
def equalizedHistogram(inImage,nbins):
	#No se porque si no llamo a la imagen aqui no me acepta
	inImage = cv2.imread('circles1.png',cv2.IMREAD_GRAYSCALE)
	width,height=inImage.shape

	x=np.linspace(0,nbins-1,num=nbins,dtype=np.uint8)
	y=np.zeros(nbins)
	#y1=np.zeros(nbins)
	modified=np.zeros(inImage.shape,inImage.dtype)

	for w in range (width):
		for h in range (height):
			v=inImage[w,h]
			y[v]=y[v]+1;

	k=(nbins-1)/(height*width)
	temp=0

	for w in range (width):
		for h in range (height):
			for z in range (inImage[w,h]):
				temp=temp+y[z]

			modified[w,h]=k*temp
			temp=0

	return modified


#filterImage filtrado espacial mediante convolucion
def filterImage(inImage,kernel):
	#dimensiones
    (iFilas,iColumnas)=inImage.shape[:2]
    (kH,kW)=kernel.shape[:2]
    #cambiar aqui
    pad=(kW)//2
    #imagen para hacer cambios
    inImage=cv2.copyMakeBorder(inImage,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    #salida
    outImage=np.zeros((iFilas,iColumnas),dtype="float32")
    #iteraciones
    for y in np.arange(pad,iFilas+pad):
        for x in np.arange(pad,iColumnas+pad):
    		#pantalla
            tam=inImage[y-pad:y+pad+1,x-pad:x+pad+1]
            #convolucion
            k=(tam*kernel).sum()
            outImage[y-pad,x-pad]=k

    outImage=rescale_intensity(outImage,in_range=(0,nbins-1))
    outImage=(outImage*(nbins-1)).astype("uint8")

    return outImage

def gaussKernel1D(sigma):

    N = int(2 * np.floor(3*sigma) + 1)
    kernel = []
    medio = int(np.floor((N/2) ))
    for x in np.arange(0, N):
    	#formula transparencias
        gaussian_index = (x-medio)
        exp=-pow(gaussian_index,2)/2/pow(sigma,2)
        fraction=1/(np.sqrt(2*np.pi)*sigma)
        kernel2=fraction*pow(np.e,exp)
        kernel.append(kernel2)

    return np.asarray(kernel)




def gaussianFilter(inImage,sigma):

	kernel=gaussKernel1D(sigma)
	#creo que se podria hacer la tarspuesta del kernel y luego el filtrado
	#convolucion de la imagen
	outImage1=cv2.filter2D(inImage,-1,kernel)
	#convolucion de la imagen con kernel traspuesto
	outImage1=cv2.filter2D(outImage1,-1,np.transpose(kernel))

	
	return outImage1


def medianFilter(inImage,filterSize):

    temp=[]
    center=filterSize // 2
    outImage=[]
    outImage=np.zeros((len(inImage),len(inImage[0])))

    for i in range(len(inImage)):
        for j in range(len(inImage[0])):
            for z in range(filterSize):
                if i + z - center < 0 or i + z - center > len(inImage) - 1:
                    for c in range(filterSize):
                        temp.append(0)
                else:
                    if j + z - center < 0 or j + center > len(inImage[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filterSize):
                            temp.append(inImage[i + z - center][j + k - center])

            temp.sort()
            outImage[i][j]=temp[len(temp) // 2]
            temp=[]

    return outImage

def highBoost(inImage,A,method,param):
	#condiciones
    if (method is None) or (param is None):
        return inImage

    if method == 'gaussian' :
        #outImage = gaussianFilter(inImage,param) #use las predeterminadas por cosas de tipos pero creo que se p
        outImage = cv2.GaussianBlur(inImage,(param,param),0) #use las predeterminadas por cosas de tipos pero creo que se p
    else :
    	if method == 'median' :
    		#outImage = medianFilter(inImage,int(param))
        	outImage = cv2.medianBlur(inImage,int(param))
    	else:
        	return None

    if A >= 0:
        #Si es positivo guardamos informacion de la imagen
        inImage = np.multiply(inImage,A).astype(np.uint8)
        return np.subtract(inImage,outImage)
    else:
        return outImage


#Aviso: en cada sitio que busco los valores de los arrays son distintos,en caso de duda cambiar los valores
def robertsOP():
    return np.array([[-1,0],[0,1]]),np.array([[0,-1],[1,0]])

def centralOP():
    return np.array([[-1,0,1]]),np.array([[-1],[0],[1]])

def prewittOP():
    return np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

def sobelOP():
    return np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),np.array([[-1,-2,-1],[0,0,0],[1,2,1]])


def gradientImage(inImage,operator):

    if operator=='Roberts':
        x,y=robertsOP()
    else:
    	if operator=='CentralDiff':
    		x,y=centralOP()
    	else:
    		if operator=='Sobel':
    			x,y=sobelOP()
    		else:
    			if operator=='Prewitt':
    				x,y=prewittOP()
    			else:
        			return print("Inexistente")

    gx=cv2.filter2D(inImage,-1,x)
    gy=cv2.filter2D(inImage,-1, y)




    #ejemplo de uso aunque entedemos que no es necesario 
 	#roberts_cross_v = np.array( [[1, 0 ],
 	#                             [0,-1 ]] )
  
	#roberts_cross_h = np.array( [[ 0, 1 ],
 	#							  [ -1, 0 ]] )
  
	# img = cv2.imread("input.webp",0).astype('float64')
	# img/=255.0
	# vertical = ndimage.convolve( img, roberts_cross_v )
	# horizontal = ndimage.convolve( img, roberts_cross_h )
  
	# edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
	# edged_img*=255
	# cv2.imwrite("output.jpg",edged_img)

    return [gx,gy]

def dilate(inImage,SE,center):

    h,w = inImage.shape
    (x, y) = SE.shape
    outImage = np.zeros(image.shape)

    return outImage


def erode(inImage,SE,center):

    h,w = inImage.shape
    (x, y) = SE.shape
    outImage = np.zeros(image.shape)

    temp = SE.sum()

    for i in range(0, h):
        for j in range(0, w):
            if outImage[i, j]!=temp:
                outImage[i][j]=0
            else:
                outImage[i][j]=1

    return outImage



def closing(inImage,SE,center):

    #primero dilate y luego erode
    outImage=cv2.dilate(inImage,SE,iterations=1)
    outImage2=cv2.erode(outImage,SE,iterations=1)

    return outImage2

def opening(inImage,SE,center):

    #primero erode y luego dilate
    outImage=cv2.erode(inImage,SE,iterations=1)
    outImage2=cv2.dilate(outImage,SE,iterations=1)

    return outImage2




inImage=cv2.imread('circles1.png')
inImage_grey = cv2.imread('circles1.png',cv2.IMREAD_GRAYSCALE)
#########FIRST FUNCTION####################

##adjustIntensity example 
# outImage1=adjustIntensity(inImage,[], [0.3, 0.7])
# cv2.imshow('original',inImage)
# cv2.imshow('outImage1',outImage1)
# compara_histogramas(inImage,outImage1)

#########SECOND FUNCTION####################
##equalizeIntensity example 

# outImage2=equalizedHistogram(inImage,nbins)
# cv2.imshow("equalizedHistogram", np.hstack((inImage_grey, outImage2)))
# im = array(inImage)
# hist(im.flatten(),nbins)
# show()
# im2 = array(outImage2)
# hist(im2.flatten(),nbins)
# show()


#########FILTER FUNCTION####################
##filter_image example 

# pokemon=cv2.imread('3d.png')
# gris=cv2.cvtColor(pokemon,cv2.COLOR_BGR2GRAY)
# laplacian = np.array((
#     [0, 1, 0],
#     [1, -4, 1],
#     [0, 1, 0]), dtype="int")

# outImage1=cv2.filter2D(gris,-1,laplacian)
# outImage2=filterImage(gris,laplacian)
# cv2.imshow('filtro predeterminado',outImage1)
# cv2.imshow('original',gris)
# cv2.imshow('mi filtro',outImage2)

#########KERNEL1D FUNCTION####################

# salida=gaussKernel1D(2)
# print(salida)


#########GAUSSIAN FUNCTION####################
# pokemon=cv2.imread('3d.png')
# outImage1=gaussianFilter(pokemon,2)
# outImage2=cv2.GaussianBlur(pokemon,(7,7),0)
# cv2.imshow("gaussian", np.hstack((pokemon, outImage1,outImage2)))

#########MEDIAN FUNCTION####################
# lisa = cv2.imread('lisa.png')
# img_median = cv2.medianBlur(lisa, 5)

# cv2.imshow("median", np.hstack((lisa, img_median)))
# img = Image.open("lisa.png").convert("L")
# arr = np.array(img)
# removed_noise = medianFilter(arr, 3) 
# img = Image.fromarray(removed_noise)
# img.show()

#########HIGH FUNCTION####################
# pokemon=cv2.imread('3d.png')
# gris=cv2.cvtColor(pokemon,cv2.COLOR_BGR2GRAY)

# method1='gaussian'
# method2='median'
# outImage1=highBoost(gris,0.5,method2,1.3)
# outImage2=highBoost(gris,2,method1,3)
# cv2.imshow("highboost", gris)
# cv2.imshow("highboost2", outImage1)
# cv2.imshow("highboost3", outImage2)



#########GRADIENT FUNCTION####################

# pokemon=cv2.imread('3d.png')
# gris=cv2.cvtColor(pokemon,cv2.COLOR_BGR2GRAY)
# out=gradientImage(gris,'Roberts')
# out2=gradientImage(gris,'Sobel')
# out3=gradientImage(gris,'Prewitt')
# out4=gradientImage(gris,'CentralDiff')
# print(out)




#########OPENING Y CLOSING FUNCTION####################
# j=cv2.imread('j.png')

# kernel = np.ones((5, 5), np.uint8)
# img_erosion = cv2.erode(j, kernel, iterations=1)
# img_dilation = cv2.dilate(j, kernel, iterations=1)
# im=erode(j,kernel,[])

# cv2.imshow('Input', j)
# cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Erosion', im)

# cv2.imshow('Dilation', img_dilation)

# closing = cv2.morphologyEx(j,cv2.MORPH_CLOSE,kernel)
# opening = cv2.morphologyEx(j, cv2.MORPH_OPEN, kernel)

# closing2 = closing(j,kernel,[])
# opening2 = opening(j,kernel,[])

# cv2.imshow('closing',closing2)
# cv2.imshow('opening',opening2)

cv2.waitKey(0)
