import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def insert_ones(X):
    ones = np.ones([X.shape[0],1])
    return np.concatenate((ones,X) , axis=1)

def transposta(matriz):
    if(len(matriz.shape) > 1):
        return matriz.T
    else:
        return np.reshape(matriz, (matriz.shape[0], 1))

def controlEntropy(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if(z[i][j] == 1):
                z[i][j] = 0.9999
            if(z[i][j] == 0 or np.isnan(z[i][j])):
                z[i][j] = 0.00001
    return z


def matriz_confusao(ypred, y):
    tp = 0 #Verdadeiro positivo (true positive — TP)
    fp = 0 #Falso positivo (false positive — FP)
    tn = 0 #Falso verdadeiro (true negative — TN)
    fn = 0 #Falso negativo (false negative — FN)
    for i in range(len(ypred)):
        if(ypred[i] == 1 and y[i] == 1):
            tp = tp +1
        if(ypred[i] == 1 and y[i] == 0):
            fp = fp +1
        if(ypred[i] == 0 and y[i] == 0):
            tn = tn +1
        if(ypred[i] == 0 and y[i] == 1):
            fn = fn +1
    matriz = [[tp, fp],[fn, tn]]
    return matriz
    
class Neural_Network(object):
    
    interacoes = 5000
    alpha = 0.005
    
    def __init__(self, X, y, f_ativacao = 'sigmoid', neuronioSize = 3, f_hide='sigmoid'): 
        self.f_ativacao = f_ativacao
        
        #Define Hyperparameters
        self.inputLayerSize = X.shape[1]
        self.outputLayerSize = y.shape[1]
        self.hiddenLayerSize = neuronioSize
        
        self.W1 = np.random.randn(self.inputLayerSize+1,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize+1,self.outputLayerSize)
        
        print('W1')
        print(self.W1)
        print('W2')
        print(self.W2)
        
        if(self.f_ativacao == 'sigmoid'):
            self.custo = self.erro_quadratico
            self.funcao_ativacao_saida = self.sigmoid
            self.funcao_derivative_saida = self.sigmoidDerivada
        if(self.f_ativacao == 'softmax'):
            self.custo = self.erro_entropia_cruzada
            self.funcao_ativacao_saida = self.softmax
        if(f_hide == 'sigmoid'):
            self.hide_function = self.sigmoid
            self.hide_derivative = self.sigmoidDerivada
        if(f_hide == 'relu'):
            self.hide_function = self.relu
            self.hide_derivative = self.reluDerivative
            
            
    #melhoria do alpha
    
    #Util alfa
    def custo_relacao_w(self, X,W, camada):    
        if(camada == 1):
            z2 = np.dot(X, W)
            a2 = self.hide_function(z2)
            a2 = insert_ones(a2)
        else:
            z2 = np.dot(X, self.W1)
            a2 = self.hide_function(z2)
            a2 = insert_ones(a2)
            
        if(camada == 2):            
            z3 = np.dot(a2, W)
            if(self.f_ativacao == 'sigmoid'):
                yHat = self.sigmoid(z3) 
            if(self.f_ativacao == 'softmax'):
                yHat = self.softmax(z3)
        else:
            z3 = np.dot(a2, self.W2)
            if(self.f_ativacao == 'sigmoid'):
                yHat = self.sigmoid(z3) 
            if(self.f_ativacao == 'softmax'):
                yHat = self.softmax(z3)
        return self.custo(yHat)
    
    def razao_aurea(self, X, W, b, grad, camada):
        a = 0
        tol = 0.03
        r = (math.sqrt(5)-1)/2
        alfa = r*a+(1-r)*b
        beta = (1-r)*a+r*b
        w_alfa = W - alfa*grad 
        y1 = self.custo_relacao_w(X, w_alfa, camada)
        w_beta = W - beta*grad 
        y2= self.custo_relacao_w(X, w_beta, camada)
        while (b-a) >= tol:
            if(y1 > y2):
                a = alfa
                alfa = beta
                y1 = y2
                beta = (1-r)*a+r*b
                w_beta = w_beta - beta*grad 
                y2= self.custo_relacao_w(X, w_beta, camada)
            else:
                b = beta
                beta = alfa
                y2 = y1
                alfa = r*a+(1-r)*b
                w_alfa = w_alfa - alfa*grad 
                y1 = self.custo_relacao_w(X, w_alfa, camada)
        return (beta+alfa)/2
        
    
    #Funcões de Ativação
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        z = controlEntropy(z)
        return 1/(1+np.exp(-z))
    
    def sigmoidDerivada(self, z):
        #Derivative of sigmoid function 
        return self.sigmoid(z)*(1 - self.sigmoid(z))   

    def relu(self, X):
        return np.maximum(0,X)

    def reluDerivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def softmax(self, z):
        z = controlEntropy(z)
        return controlEntropy(np.exp(z) / np.sum(np.exp(z)))
    
    def maxvalor(self, y):
        eps = 1e-4
        ylinhas, ycolunas = y.shape
        s = np.zeros((ylinhas, ycolunas))
        for i in range(ylinhas):
            for j in range(ycolunas):
                if(np.amax(y[i]) == y[i][j]):
                    s[i][j] = 1
                else:
                    s[i][j] = 0
        return s 
                           
    #Funções de Custo
    
    def erro_quadratico(self, yHat):
        #Calculo do custo       
        erro = 1/2*(self.y - yHat)**2
        Erro_total = np.sum(erro, axis = 1)
        return sum(Erro_total)
    
    
    def erro_entropia_cruzada(self, yHat):
        entropy = self.y*np.log(yHat)
        return  -1*np.sum(entropy, axis = 0)
    
    
    #Propagação
    def forward(self, X):        
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.hide_function(self.z2)
        self.a2 = insert_ones(self.a2)
        
        self.z3 = np.dot(self.a2, self.W2)
        if(self.f_ativacao == 'sigmoid'):
            yHat = self.sigmoid(self.z3) 
        if(self.f_ativacao == 'softmax'):
            yHat = self.softmax(self.z3)

        return yHat
    
    #Retro Propagação   
    def costFunctionDerivada(self, X, y):

        if(self.f_ativacao == 'sigmoid'):
            delta3 = np.multiply(-(y-self.yHat), self.funcao_derivative_saida(self.z3))
        if(self.f_ativacao == 'softmax'):
            z = controlEntropy(self.yHat)
            z = np.round(z)
            delta3 = (z - y)

        dJdW2 = np.dot(transposta(self.a2), delta3)
        
        remover_w2_bias = np.delete(self.W2, (0), axis=0)

        delta2 =  np.dot(delta3, transposta(remover_w2_bias))*self.hide_derivative(self.z2)

        dJdW1 = np.dot(X.T, delta2)  

        dJdX = np.dot(delta2, transposta(self.W1))

        dJdX = np.delete(dJdX, (0), axis=1)
        
        return dJdW1, dJdW2, dJdX
    
    def updateWeight(self,X, dJdW1, dJdW2):
        self.alpha = self.razao_aurea(X, self.W2, 1, dJdW2, 2)
        self.W2 = self.W2 - self.alpha*dJdW2

        self.alpha = self.razao_aurea(X, self.W1,1, dJdW1, 1)
        self.W1 = self.W1 - self.alpha*dJdW1

        
    def predict(self, X):
        X = insert_ones(X)
        return self.forward(X)
        
    def treinar(self, X, y):        
        X = insert_ones(X)
        self.y = y
        iteration_counter = 0
        erro = 99999
        eps = 1e-2
        self.erros = np.zeros(self.interacoes)
        while abs(erro) > eps and iteration_counter < self.interacoes: 
            self.yHat = self.forward(X)
            erro = self.custo(self.yHat)
            dJdW1, dJdW2, dJdX = self.costFunctionDerivada(X, y) 
            self.erros[iteration_counter] = erro
            self.updateWeight(X, dJdW1, dJdW2)
            iteration_counter += 1
        print('relatorio')
        print('erro:', erro)
        print('interações:',iteration_counter,'/',self.interacoes)
        print('W1:')
        print(self.W1)
        print('W2:')
        print(self.W2)
        print()

    def backpropagation(self):
        dJdW1, dJdW2, dJdX = self.costFunctionDerivada(self.X, self.y) 
        self.updateWeight(self.X, dJdW1, dJdW2)
        return dJdX
        
    def fit(self, X, y):
        X = insert_ones(X)
        self.X = X
        self.y = y
        self.yHat = self.forward(X)
        erro = self.custo(self.yHat)
        return erro, self.yHat