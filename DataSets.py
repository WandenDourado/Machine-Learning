import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def diabetes_dataset():
    df=pd.read_csv("dataset_train.txt", sep="\t", header=None)
    df_norm = (df - df.mean()) / df.std()
    treino_x = df_norm[[0,1,2,3,4,5,6,7]].to_numpy()
    treino_y = df[[8]].to_numpy()
    df_teste=pd.read_csv("dataset_teste.txt", sep="\t", header=None)
    df_norm_test = (df_teste - df_teste.mean()) / df_teste.std()
    df_norm_test.head()
    teste_x = df_norm_test[[0,1,2,3,4,5,6,7]].to_numpy()
    teste_y = df_teste[[8]].to_numpy()
    return treino_x, treino_y, teste_x, teste_y

def hebatite_dataset():
    df=pd.read_csv("hepatitis.data", sep=",", header=None)
    colunas = df.shape[1] - 1
    for i in range(colunas):
        df[i] = valores_faltantes(df[i])
        df[i] = pd.to_numeric(df[i])
    df_yes = df.loc[df[19] == 1]
    df_no = df.loc[df[19] == 2]
    #dividir dataframe por classe
    qtdTreino = int(df_yes.shape[0]*2/3)
    df_yes_treino = df_yes[0:qtdTreino]
    df_yes_teste = df_yes[qtdTreino:]

    qtdTreino = int(df_no.shape[0]*2/3)
    df_no_treino = df_no[0:qtdTreino]
    df_no_teste = df_no[qtdTreino:]
    #Juntas os dataframes de treino e teste
    frames_treino = [df_yes_treino, df_no_treino]
    frames_teste = [df_yes_teste, df_no_teste]
    df_treino = pd.concat(frames_treino)
    df_teste = pd.concat(frames_teste)

    df_treino = df_treino.reset_index(drop=True)
    df_teste = df_teste.reset_index(drop=True)
    treino_x = (df_treino[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] - df_treino[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].mean()) / df_treino[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].std()
    teste_x = (df_teste[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] - df_teste[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].mean()) / df_teste[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].std()
    treino_y = df_treino[[19]] 
    teste_y = df_teste[[19]] 
    treino_x = treino_x.to_numpy()
    treino_y = treino_y.to_numpy()
    teste_x = teste_x.to_numpy()
    teste_y = teste_y.to_numpy()
    treino_y = np.where(treino_y==1, 0, treino_y) 
    treino_y = np.where(treino_y==2, 1, treino_y) 
    teste_y = np.where(teste_y==1, 0, teste_y) 
    teste_y = np.where(teste_y==2, 1, teste_y) 
    return treino_x, treino_y, teste_x, teste_y
    


def valores_faltantes(vetor):
    linha = vetor.shape[0]
    qtdNaoFaltantes = 0
    somaValoresNaoFaltantes = 0
    for i in range(linha):
        if(vetor[i] != '?'):
            qtdNaoFaltantes = qtdNaoFaltantes + 1
            somaValoresNaoFaltantes = somaValoresNaoFaltantes + float(vetor[i])
    for i in range(linha):
        if(vetor[i] == '?'):
            vetor[i] = int(somaValoresNaoFaltantes/qtdNaoFaltantes)
    return vetor
