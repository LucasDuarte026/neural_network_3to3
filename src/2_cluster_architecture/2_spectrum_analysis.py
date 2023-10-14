import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_data():
    
    plt.figure(10)

   
    x12_spectrum = np.fft.fft(x12.predicts)
    frequencias = np.fft.fftfreq(len(x12.predicts), d=1/0.05)
    
    module = x12_spectrum*np.conj(x12_spectrum)
    
    plt.loglog(frequencias*len(x12_spectrum),module, label='x12', alpha=0.5)


    plt.plot(x_aux, y_aux, label='aux line -5/3', alpha=0.5)


    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Espectro da linha temporal dos 4/8/10/12x16")
    plt.legend()


def show_spectrum(big_data_df, sonic):

    plt.figure(0)

    velocity_hotfilm_spectrum = np.fft.fft(big_data_df.velocity_hotfilm)
    frequencias = np.fft.fftfreq((big_data_df.velocity_hotfilm.size), d=1/2000)
    
    teste = velocity_hotfilm_spectrum*np.conj(velocity_hotfilm_spectrum)
    
    plt.loglog(frequencias,teste, label='velocity_hotfilm', alpha=0.5)
    plt.plot(x_aux, y_aux, label='aux line -5/3', alpha=0.5)
    
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Espectro da linha temporal dos velocity_hotfilm")
    plt.legend()

    # plt.figure(1)

    # velocity_hotfilm_spectrum = np.fft.fft(sonic.velocity_sonic)
    # frequencias = np.fft.fftfreq(len(sonic.velocity_sonic), d=1/20)

    # plt.loglog(frequencias, np.abs(velocity_hotfilm_spectrum), label='velocity_sonic', alpha=0.5)
    # plt.plot(x_aux, y_aux, label='aux line -5/3', alpha=0.5)

    # plt.xlabel("Frequency")
    # plt.ylabel("Amplitude")
    # plt.title("Espectro da linha temporal dos velocity_sonic")
    # plt.legend()


# ler todos os dados
big_data_df = pd.read_csv(
    './big_data_df.csv', sep=',')
sonic = pd.read_csv(
    './sonic_df.csv', sep=',')
# x4 = pd.read_csv('/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/Dados_gerados/levantamento_dados/resultado_4x16/resultado_predict_4x16.csv', sep=',')
# x8 = pd.read_csv('/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/Dados_gerados/levantamento_dados/resultado_8x16/resultado_predict_8x16.csv', sep=',')
# x10 = pd.read_csv('/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/cluster_architecture/Dados_gerados/levantamento_dados/resultado_10x16/resultado_predict_10x16.csv', sep=',')
x12 = pd.read_csv('./resultado_predict_12x16.csv', sep=',')


# Inserindo a reta com -5/3 de reta angular f(x) = ang*x^expo

x_aux = np.linspace(0, 1000, 10000, endpoint=False)
ang, expo = 1000000, -5/3
y_aux = ang*(x_aux ** expo)


show_spectrum(big_data_df, sonic)       # Está fazendo o plot com os dados originais base
# show_data()                             # está fazendo teste com o dados do x12 processados
plt.show()
