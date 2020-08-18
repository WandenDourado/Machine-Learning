import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self,x,y):
        elementos,variaveis = x.shape
        x_bias = np.ones((elementos,1))
        x = np.append(x_bias,x,axis=1)
        x_transpose = np.transpose(x)
        x_transpose_dot_x = x_transpose.dot(x)
        parte_1 = np.linalg.inv(x_transpose_dot_x)
        parte_2 = x_transpose.dot(y)
        theta =parte_1.dot(parte_2)
        self.theta = theta

    def predict(self, x):
        elementos,variaveis = x.shape
        x_bias = np.ones((elementos,1))
        x = np.append(x_bias,x,axis=1)
        return np.round(np.dot(x, self.theta))
