import pyopencl as cl
import cv2 as cv
import numpy  as np
import sys
import time
import re

def SetContoursVector(mat):
    ret,thresh = cv.threshold(mat,127,255,0)
    contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    return contours

def splitAxis(contourList,i):
    contours2 = np.empty((len(contourList[i]),2),int)
    for j in range(len(contourList[i])):
      contours = str(contourList[i][j]).split(sep = ' ')
      k = 0
      while( k < len(contours)):
          if(contours[k] == ''):
              contours = np.delete(contours,k)
              k = k-1
          k = k+1  

      if(len(contours) == 3):
          contours[0] = contours[1]
          contours[1] = contours[2]    
      contours2[j][0] = re.sub('\[','',contours[0])
      contours2[j][1] = re.sub('\]','',contours[1])
    return contours2

def middlePoints(contours):
    pixels = np.zeros((len(contours)),dtype = np.float64)
    for i in range(len(contours)):
        contourAxis = splitAxis(contours,i)
        for j in range(len(contourAxis)):
            current = contourAxis[j][0]
            for k in range(len(contourAxis)):
                if(current == contourAxis[k][0] and j!=k):
                    pixels[i] = pixels[i] + (abs(contourAxis[j][1]-contourAxis[k][1])/2)
    return pixels

def ShowContours(contours,tam):
    f = open('output\\contours.txt', 'w')
    totalSuperficieParticula = 0
    for i in range(len(contours)):
        tamanoParticula = contours[i]*8000/tam
        tamanoParticula = round(tamanoParticula,3)
        totalSuperficieParticula = totalSuperficieParticula +tamanoParticula
        particle = 'CONTORNO '+str(i)+': con '+str(tamanoParticula)+' milímetros cuadrados'
        f.write(particle+ '\n')

    f.write('Número de partículas: '+str(len(contours)) + '\n')
    totalSuperficieParticula = round(totalSuperficieParticula,3)
    f.write('Total superficie partícula: ' + str(totalSuperficieParticula) + ' milímetros cuadrados \n')
    f.write('Porcentaje partículas en imagen: ' + str(round(totalSuperficieParticula*100/8000,3)) + '%\n')

def DeleteSmallObjects(contours):
    delete = []
    for i in range(len(contours)):
        if(len(contours[i]) < 55): 
                delete.append(i)

    contours2 = np.delete(contours,delete,0)
    return contours2


def settingArguments(nombre):
    f = open('files\\particles.txt',"r")
    found = 0
    a = 0
    for i in f:
        line = i.split(sep=',')
        if(nombre == line[0]):
            a = np.array([line[1],line[2],line[3]])
            break

    return a

def findParticle(number):
     f = open('files\\particles.txt', 'r')
     n = 0
     for i in f:
        line = i.split(sep=',')
        if(np.intc(number)== n):
            return line[0]
        n = n+1

     return 0
     


#----------------------------------main program------------------------------------------#

#read image and set standard size 
image = cv.imread(sys.argv[2])
if(sys.argv[2] == ""):
    print("Error, debe seleccionar una ruta o la ruta seleccionada no existe.")
    exit()



nombre = findParticle(sys.argv[1])
if(nombre == 0):
    print("Error, la partícula especificada no existe o no se ha especificado")
    exit()


#Standard size: 4056*3040
tam =4056*3040 
size = (4056,3040)

pixels = image.shape[0]*image.shape[1]

if(pixels != tam):
    print("Tamaño de imagen distinto al adecuado, redimensionando...")
    image = cv.resize(image,size)


#gray scale transforming

gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

#-----------------------------------OpenCL-----------------------------------------------#

#available platforms
print(cl.get_platforms())

#choose platform and device

platform = cl.get_platforms()[0]
gpu = cl.get_platforms()[0].get_devices()
print(gpu)

#creating context

context = cl.Context(
                        dev_type = cl.device_type.ALL,
                        properties= [(cl.context_properties.PLATFORM, platform)])


#creating command queue

queue = cl.CommandQueue(context)


#Creating buffers

mf = cl.mem_flags

imageBuffer = gray.astype(np.uint8)

#image buffer and other arguments
a_g = cl.Buffer(context,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imageBuffer)

a = settingArguments(nombre)

minBtm = np.intc(a[0])
maxBtm = np.intc(a[1])
margin = np.intc(a[2])

rows = np.intc(imageBuffer.shape[0])
cols = np.intc(imageBuffer.shape[1])

#result vector buffer
b_g = cl.Buffer(context,mf.WRITE_ONLY ,  imageBuffer.nbytes)

#creating program
src = ''.join(open('build\\scripts-3.9\\kernel.cl').readlines())

program = cl.Program(context,src).build()

#setting arguments and launching program
start = time.time()
program.btm(queue,imageBuffer.shape,None, a_g, b_g,rows,cols,maxBtm,margin,minBtm)
end = time.time()

#getting results

print('El tiempo de ejecución es: ', end-start, 'segundos')
result = np.empty_like(imageBuffer)
cl.enqueue_copy(queue,result,b_g)


#--------------------------------------------------OpenCV----------------------------------------------------#

#set contours vector
contours = SetContoursVector(result)
contours = np.array(contours,dtype = object)
contours = DeleteSmallObjects(contours)
pixels = middlePoints(contours)
ShowContours(pixels,tam)
cv.drawContours(image,contours,-1,(0,255,73),3)
print('Número de',nombre,': ',contours.shape[0])

#saving images
cv.imwrite('output\\gray.jpg',gray)
cv.imwrite('output\\result.jpg',result)
cv.imwrite('output\\contours.jpg',image)

#resize images
result = cv.resize(result,(640,480))
image = cv.resize(image,(640,480))
gray = cv.resize(gray,(640,480))

#show images
cv.imshow('micro',result)
cv.imshow('Contours',image)
cv.imshow('gray',gray)

