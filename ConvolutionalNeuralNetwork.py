import numpy as np
import ConvolutionLayer as CL
import PoolingLayer as PL
import FullyConnectedLayer as FC
import cv2
import ImageUtil as ImageUtil
import CreateData as CreateData
import Image as Image
import LayerCNN as Layer
import Log as Log
from datetime import datetime
import matplotlib.pyplot as plt

class CNN(object):

    layers = np.empty(0)
    img = cv2.imread('img/X.jpg')
    interactions = 500
    errors = np.zeros(interactions)

    def addConvolutionLayer(self, width_filter=2, height_filter=2 ,strider=1, filter_amount=5):
        convolution_layer = CL.ConvolutionLayer(width_filter, height_filter, strider, filter_amount)
        layer = Layer.Layer(convolution_layer, 'convolution')
        self.layers = np.append(self.layers, layer)

    def addPoolingLayer(self, width_filter=2, height_filter=2 ,strider=1):
        pooling_layer = PL.PoolingLayer(width_filter, height_filter, strider)
        layer = Layer.Layer(pooling_layer, 'pooling')
        self.layers = np.append(self.layers, layer)

    def addFullyLayer(self):
        fully_layer = FC.FullyConnectedLayer()
        layer = Layer.Layer(fully_layer, 'fully')
        self.layers = np.append(self.layers, layer)

    def forward(self):
        print('forward')
        for layer in self.layers:
            print(layer.name)
            if(layer.name == 'fully'):
                self.errors[self.iteration_counter], self.yHat = layer.layer.process(self.train_x, self.train_y)
            else:
                for i in range(len(self.train_x)):
                    self.train_x[i] = layer.layer.process(self.train_x[i])

    def backpropagation(self):
        print('backpropagation')
        for layer in reversed(self.layers):
            print(layer.name)
            if(layer.name == 'fully'):
                dJdx = layer.layer.backpropagation()
            else:
                filters_samples = []
                for i in range(len(self.train_x)):                    
                    if(layer.name == 'convolution'):
                        dJdx[i], filters_layer = layer.layer.backpropagation(dJdx[i])
                        filters_samples.append(filters_layer)
                        if(i+1 == len(self.train_x)):
                            layer.layer.filters = self.calculteMeanFilters(filters_samples)
                    else:
                        dJdx[i] = layer.layer.backpropagation(dJdx[i])

    def calculteMeanFilters(self, filters_samples):
        width_filter, height_filter = filters_samples[0][0].shape
        samples_size = len(filters_samples)
        filter_size = len(filters_samples[0])
        mean_filter = np.zeros((filter_size, width_filter, height_filter))
        for sample in range(samples_size):
            for filter in range(filter_size):
                for j in range(height_filter):
                    for i in range(width_filter):
                        mean_filter[filter][i,j] = mean_filter[filter][i,j] + filters_samples[sample][filter][i,j]
        for filter in range(filter_size):
                for j in range(height_filter):
                    for i in range(width_filter):
                        mean_filter[filter][i,j] = mean_filter[filter][i,j]/samples_size
        return mean_filter
 

    def predict(self):
        print('predict')
        self.createDataBase()
        self.forward()
        y_pred = np.round(self.yHat)
        corretos = (y_pred == self.train_y).sum()
        total = len(self.train_x)
        print(corretos,"/",total)
        taxa_de_acerto = corretos/total
        print("Taxa de acerto %.2f " % (taxa_de_acerto*100))
        fig, ax = plt.subplots()  
        ax.plot(np.arange(self.interactions), self.errors, 'r')  
        ax.set_xlabel('Iterações')  
        ax.set_ylabel('Custo')  
        ax.set_title('Erro vs. Epoch')
        plt.show()
        return self.yHat

    def fit(self):
        self.iteration_counter = 0
        error = 99999
        eps = 1e-2
        now = datetime.now() 
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        data_execucao = "Executado em "+dt_string+"\n"
        Log.registrar("run_error.txt", data_execucao)
        while abs(error) > eps and (self.iteration_counter+1) < self.interactions:
            self.createDataBase()
            self.forward()
            error = self.errors[self.iteration_counter]
            registro = str(self.iteration_counter)+" - "+str(error)+"\n"
            Log.registrar("run_error.txt", registro)
            self.backpropagation()
            self.iteration_counter += 1
        print('relatorio')
        print('erro:', error)
        print('interações:',self.iteration_counter,'/',self.interactions)
        print(self.iteration_counter)
            

    def createDataBase(self):
        data = CreateData.CrossValidation(50)
        dsize = (60, 60)
        self.train_y = np.zeros((len(data),1))
        self.train_x = []
        position = 0
        for sample in data:
            img_1 = cv2.imread(sample[0], cv2.IMREAD_GRAYSCALE)
            img_1 = ImageUtil.getFace(img_1)
            img_1 = ImageUtil.hogFeature(img_1)
            img_1 = cv2.resize(img_1, dsize)

            img_2 = cv2.imread(sample[1], cv2.IMREAD_GRAYSCALE)
            img_2 = ImageUtil.getFace(img_2)
            img_2 = ImageUtil.hogFeature(img_2)
            img_2 = cv2.resize(img_2, dsize)

            img = cv2.hconcat([img_1, img_2])

            pkg1 = Image.Package()
            pkg1.addImage(img)

            self.train_x.append(pkg1)
            self.train_y[position] = int(sample[2])
            position = position +1