#%%
import DataSets as biblioteca
import LinearRegression as LR
import MultiLayerPerceptron as MLP
import numpy as np
import matplotlib.pyplot as plt
import ConvolutionalNeuralNetwork as CNN

def testar_modelo():
    new_cnn = CNN.CNN()
    new_cnn.addConvolutionLayer(width_filter=4, height_filter=4 ,strider=1, filter_amount=2)
    new_cnn.addPoolingLayer()
    #new_cnn.addConvolutionLayer(width_filter=2, height_filter=2 ,strider=1, filter_amount=5)
    #new_cnn.addPoolingLayer()
    new_cnn.addFullyLayer()
    new_cnn.fit()
    new_cnn.predict()

testar_modelo()