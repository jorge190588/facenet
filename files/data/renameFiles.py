import os
import random

dir_path =  os.getcwd()

baseDir = "..\\lfw\\imagenesDeEntrada\\"
outputDif =os.path.abspath(os.path.join(dir_path,"..","lfw\\imagenesOrdenadas"))
print('outputDir',outputDif)

for name in os.listdir(baseDir):
    if name.find(".txt") == -1:
        path, dirs, files = next(os.walk(baseDir+name))
        
        directory = outputDif+'\\'+name
        try:
            os.stat(directory)
            print ('directory already exist ',directory)
        except:
            os.mkdir(directory)  
            print('create ',directory)
            
        indice = 1
        for file in os.listdir(baseDir + name):
            oldFile = dir_path+'\\'+baseDir+name+'\\'+ file
            newFile= outputDif+'\\'+name+'\\'+name+'_'+ '%04d' % int(indice) +'.jpg'
            print('imagen original ',oldFile,'newFile',newFile)
            indice =indice+1
            os.rename(oldFile, newFile)
        