import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
    Como usar:

$ python3 grapher.py {diretório com a pasta com os dados} nome_do_arquivo
exemplo:
$ python3 grapher.py levantamento_de_dados/ 14x16

'''
DIR = sys.argv[1]
SERIE = sys.argv[2]

input_df_name = "voltage"
output_df_name = "velocity_sonic"


# Generalizando código para ser usado em mais de uma máquina
dir_base = f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/2_cluster_architecture/Dados_gerados/{DIR}'


def show_graphs(data, predictions, see_train_loss, see_val_loss):
    see_train_loss = see_train_loss.drop(0)
    see_val_loss = see_val_loss.drop(0)

    # Showing data

    shown = predictions
    print('\n\n\n\nPREVISTO\n\n\n\n', predictions)
    shown = shown.assign(original=data[output_df_name])
    print("shown\n")
    print(shown)

    current_perdiod = float(shown['time'][1]-shown['time'][0])
    N = len(predictions.predicts)
    
    
    plt.figure(0)
    # Plotting both the curves simultaneously
    if (output_df_name == "velocity_sonic"):
        plt.plot(predictions.time, predictions.velocity_sonic,
                 color='r', label='data')
    elif (output_df_name == "velocity_hotfilm"):
        plt.plot(data.time, data.velocity_hotfilm, color='r', label='data')

    plt.plot(predictions.time, predictions.predicts,
             color='g', label='processed')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset")
    plt.legend()

    
    plt.figure(1)
    x_aux = np.linspace(0, N*10, 10000, endpoint=False)
    ang, expo = 1, -5/3
    y_aux = ang*(x_aux ** expo)

    spectrum = np.fft.fft(predictions.predicts)
    frequencias = np.fft.fftfreq(N, d=current_perdiod)
    module = spectrum*np.conj(spectrum)

    plt.loglog(frequencias*N, module/(max(module)),label=f'espectro do {SERIE}', alpha=0.8)
    plt.plot(x_aux, y_aux, label='aux line -5/3',alpha=0.8, color='orange')

    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(f"Espectro da linha temporal do {SERIE}")
    plt.legend()

    plt.figure(2)
    plt.title("Evolução do erro de treino ao longo do tempo")
    print('see_train_loss\n', see_train_loss)
    plt.plot(see_train_loss['time'],
             see_train_loss['error_train'], color='g', label='train')
    plt.xlabel("Interação")
    plt.legend()

    plt.figure(3)
    plt.title("Evolução do erro da validação ao longo do tempo")
    plt.plot(see_val_loss['time'], see_val_loss['error_val'],
             color='r', label='validation')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()
    
    
    plt.show()


print('\n\n\t | Diretório dos dados e nome do arquivo: || ', DIR, SERIE, '|\n')

# ler todos os dados
data = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/2_cluster_architecture/dados_comAleatoriedade.csv', sep=',')
predict = pd.read_csv(
    f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/2_cluster_architecture/Dados_gerados/{DIR}/resultado_{SERIE}/resultado_predict_{SERIE}.csv', sep=',')
train = pd.read_csv(
    f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/2_cluster_architecture/Dados_gerados/{DIR}/resultado_{SERIE}/resultado_train_{SERIE}.csv', sep=',')
val = pd.read_csv(
    f'/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/2_cluster_architecture/Dados_gerados/{DIR}/resultado_{SERIE}/resultado_val_{SERIE}.csv', sep=',')
print(data)

show_graphs(data.head(len(predict)), predict, train, val)
