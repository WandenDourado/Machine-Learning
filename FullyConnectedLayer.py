import cv2
import numpy as np
import MultiLayerPerceptron as MLP
import Image as Image
from datetime import datetime
import Log as Log

class FullyConnectedLayer(object):

    def __init__(self, neuronioSize=500):
        self.neuronioSize = neuronioSize
        self.model_create = False

    def create_vector_x(self, package):
        x = []
        for image in package.images:
            width_image = image.width
            height_image = image.height          
            for i in range(width_image):
                for j in range(height_image):
                    for layer in range(image.layers):
                        x.append(image.img[i,j,layer])
        return x

    def recreate_image(self, x):
        recreate = []        
        sample = -1
        x[np.isnan(x)] = 0

        for package in self.amostras:
            sample = sample + 1
            vector_position = 0
            index = -1
            pkg = Image.Package()            
            recreate.append(pkg)
            for image in package.images:
                width_image = image.width
                height_image = image.height             
                recreate[sample].addImage(np.zeros((width_image,height_image,image.layers)))  
                index = index + 1
                for i in range(width_image):
                    for j in range(height_image):
                        for layer in range(image.layers):
                            new_pixel = x[sample][vector_position]
                            recreate[sample].setPixelImageByLayer(index, new_pixel, i,j, layer)
                            vector_position = vector_position+1      
  
        return recreate

    def backpropagation(self):
        dJdX = self.model.backpropagation()
        dJdX = self.recreate_image(dJdX)
        return dJdX

    def process(self, train_x, train_y):
        self.amostras = train_x
        self.big_x = []
        self.amostras_size = len(train_x)
        for i in range(self.amostras_size):
            self.x = self.create_vector_x(train_x[i])
            self.big_x.append(self.x)
        train_x = np.array(self.big_x) 
        print(train_x)
        train_y = np.reshape(train_y, (train_y.shape[0], 1))
        if(self.model_create == False):
            now = datetime.now() 
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            data_execucao = "Executado em "+dt_string+"\n"
            Log.registrar("run_config.txt", data_execucao)
            info_modelo = "MLP -> f_atv = softmax, numero_de_neuronios=  "+str(self.neuronioSize)+"\n"
            Log.registrar("run_config.txt", info_modelo)
            self.model = MLP.Neural_Network(train_x,train_y, f_ativacao = 'sigmoid', neuronioSize=self.neuronioSize, f_hide='sigmoid')
            self.model_create = True
        error, yHat = self.model.fit(train_x, train_y)
        return error, yHat