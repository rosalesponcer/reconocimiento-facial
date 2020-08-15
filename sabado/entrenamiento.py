import cv2
import os
import numpy as np

dataPath = os.getcwd()+'\\data'

peopleList = os.listdir(dataPath)

print('Lista de empleados: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personalPath = dataPath+'/'+nameDir
    print('Leyendo imagenes ...')

    for fileName in os.listdir(personalPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personalPath+'/'+fileName, 0))
        image = cv2.imread(personalPath+'/'+fileName, 0)
        # cv2.imshow('image',image)
        # cv2.waitKey(10)

    label = label + 1

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print("Entrenando...")

face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloEigenFaces.xml')
print('Modelo almacenado...')
