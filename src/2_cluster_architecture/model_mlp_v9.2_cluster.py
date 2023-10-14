import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

'''
-- -- -- -- -- -- Como usar o algorítmo -- -- -- -- -- -- 

    -> Caso 1 - Treinar e rodar a rede com os mesmos dados:

$ python3 model_mlp_v{VERSION}_cluster.py {nome final do arquivo exportado} {hidden_layers} {hidden_size}
exemplo:
python3 model_mlp_v{VERSION}_cluster.py 12x16 12 16


    -> Caso 2 - Rodar no banco de dados com modelo salvo :

$ python3 model_mlp_v{VERSION}_cluster.py "run"


'''

'''
    * Esta versão contém o upgrade com o
    @author lucas
    *
    *
    *
    *
'''
__author__ = "Lucas Sales Duarte"
__email__ = "lucassalesduarte026@gmail.com"
__status__ = "Production"

EPOCHS = 1000
input_size = 1
output_size = 1

hidden_layers = 2       # 2 comum
hidden_size = 8         # 8 comum

learning_rate = 0.01
batch_size = 32

amount = 3000      # Tamanho do dataset  |  se negativo, usará o data set todo

''' observação
O tamanho dos dados do df com o predict é amout
O tamanho dos dados de treino é igual a 3 * EPOCHS
O tamanho dos dados de validação é igual à  EPOCHS
'''


VERSION = 9.2         # controle de versionamento

# MODOS
EXPORT_DATA = True    # Exporta arquivos .csv para analizar o resultado da rede
GRAPHS = True         # Mostrar os gráficos
SAVE = False          # Salvar o modelo
GPU = 0               # 0 para uso da CPU                   | 1 para uso da GPU
REAL = 1              # 0 para TREINO COM O DF FAKE(HOT)    | 1 para TREINO COM O DF REAL (SONIC)
# "MODE"              # 0 para apenas rodar a rede (fake)   | 1 para treinar a rede


LOCAL = 1             # 0 para no cluster                   | 1 para TREINO no notebook


if (len(sys.argv) < 2):
    print("\n\nentrada errada, digite nesse formato:\n\n-- -- -- -- -- -- Como usar o algorítmo -- -- -- -- -- --\n \n\n\t-> Caso 1 - Treinar e rodar a rede com os mesmos dados:\n\n$ python3 model_mlp_v{VERSION}_cluster.py {nome final do arquivo exportado} {hidden_layers} {hidden_size}\nexemplo:\npython3 model_mlp_v{VERSION}_cluster.py 12x16 12 16\n\n\n\n\t-> Caso 2 - Rodar no banco de dados com modelo salvo :\n$ python3 model_mlp_v{VERSION}_cluster.py run\n")
    sys.exit()


elif (sys.argv[1]   == "run"):
    print("\n\n -- -- -- -- - -- -- -- ")
    print("\n\n Modo de rodar o algoritmo ")
    print("\n\n -- -- -- -- - -- -- -- ")
    MODE = 0    # pronto para rodar com modelo salvo 


elif len(sys.argv) > 3:
    hidden_layers = int(sys.argv[2])    # Quantidade de camadas
    hidden_size = int(sys.argv[3])      # Quantidade de neuronios
    MODE = 1    # pronto para treinar 
    
SERIE = sys.argv[1]       # Nome do código
# Generalizando código para ser usado em mais de uma máquina
caminho_local = '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/2_cluster_architecture'
caminho_cluster = '/home/lucasdu/algoritmo/2_cluster_architecture'
dir_base = ''
if LOCAL == 1:
    dir_base = caminho_local
elif LOCAL == 0:
    dir_base = caminho_cluster
    GRAPHS = False


# Carregamento dos dados para dentro da rede
if (MODE):
    # Carregar a rede com os dados reais
    df = pd.read_csv(
        f'{dir_base}/dados_comAleatoriedade.csv', sep=",")
    print("\n\nTreino com DataFrame dos dados do sensor sônico\n\n")
    input_df_name = "voltage"
    output_df_name = "velocity_sonic"
else:
    # Carregar a rede com os dados sintéticos
    df = pd.read_csv(
        f'{dir_base}/dadosFake_comAleatoriedade3M6.csv', sep=",")
    print("\n\nTreino com DataFrame dos dados sintéticos (gerados através da função sintética)\n\n")

    input_df_name = "voltage"
    output_df_name = "velocity_hotfilm"

# Seleção do dispositivo de processamento
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if GPU == False:
    device = 'cpu'
print(f"\tDevice de processamento: {device}\n")

# Cabeçalho
print(
    f' -- -- Tipos dos dados do df-- --  \n----------------------------\n{df.dtypes}\n----------------------------')
if amount > 0:
    df = df.head(amount)
amount = len(df)
print(
    f"\n\t __ - __ A quantidade de dados do TS a serem processados pela rede é: {amount} __ - __\n")
print(df)


'''
    Definição da classe que controla os parâmetros da arquitetura da rede
        - Número de camadas e neurônios de cada rede
        - formato de entrada e saída da rede
        - definição das funções de ativação

'''


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=hidden_size, num_hidden_layers=hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


'''
    Uso de uma classe para criar os objetos dataset para treino
        - Separados por dataset de treino e de validação
'''


class VoltageVelocityDataset(Dataset):
    def __init__(self, data):
        self.X = (torch.tensor(
            data[input_df_name].values).float().unsqueeze(1)).to(device)
        self.Y = (torch.tensor(
            data[output_df_name].values).float().unsqueeze(1)).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def export_data(df, predictions, see_train_loss, see_val_loss):
    predictions = pd.DataFrame(predictions)
    see_train_loss = pd.DataFrame(see_train_loss)
    see_val_loss = pd.DataFrame(see_val_loss)

    df_exp = df[['time', 'voltage', 'velocity_sonic']]
    df_exp = df_exp.join(predictions)
    df_exp = df_exp.rename(columns={df_exp.columns[-1]: 'predicts'})

    caminho_completo = os.path.join(dir_base, f'resultado_{SERIE}')
    if not os.path.exists(caminho_completo):
        os.makedirs(caminho_completo)

    df_exp.to_csv(
        f'{caminho_completo}/resultado_predict_{SERIE}.csv', index=False)
    see_train_loss.to_csv(
        f'{caminho_completo}/resultado_train_{SERIE}.csv', index=False)
    see_val_loss.to_csv(
        f'{caminho_completo}/resultado_val_{SERIE}.csv', index=False)

    diff_media, diff_max, diff_min = trained_info(df, predictions)
    print(
        f'\nMédia da diferença: {diff_media:6.6f}\nMáxima diferença:   {diff_max:6.6f}\nMínima diferença:   {diff_min:6.6f}\n')

    df_hyper = pd.DataFrame(
        {
            'Média': [diff_media],
            'Máximo': [diff_max],
            'Mínimo': [diff_min],
            'Epochs': [EPOCHS],
            'hidden_layers': [hidden_layers],
            'hidden_size': [learning_rate],
            'learning_rate': [learning_rate],
            'batch_size': [batch_size],
            'amount': [amount],
            'hidden_size': [hidden_size],
        })
    print('\ndf_hyper\n', df_hyper)

    df_hyper.to_csv(
        f'{caminho_completo}/hyperparameters_{SERIE}.csv', index=False)


def show_graphs(data, predictions, see_train_loss, see_val_loss):
    # Showing data
    shown = predictions
    shown = shown.assign(original=data[output_df_name])
    see_train_loss = pd.DataFrame(see_train_loss)
    see_val_loss = pd.DataFrame(see_val_loss)
    see_train_loss = see_train_loss.drop(0)
    see_val_loss = see_val_loss.drop(0)
    # pd.set_option('display.max_rows', None)
    print(shown)

    plt.figure(0)
    # Plotting both the curves simultaneously
    if (REAL):
        plt.plot(data.time, data.velocity_sonic, color='r', label='data')
    else:
        plt.plot(data.time, data.velocity_hotfilm, color='r', label='data')
    plt.plot(data.time, predictions,
             color='g', label='processed')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset")
    plt.legend()

    plt.figure(2)
    plt.title("Evolução do erro de treino ao longo do tempo")
    plt.plot(see_train_loss[see_train_loss.columns[0]], see_train_loss[see_train_loss.columns[1]],
             color='g', label='train')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()

    plt.figure(3)
    plt.title("Evolução do erro da validação ao longo do tempo")
    plt.plot(see_val_loss[see_val_loss.columns[0]], see_val_loss[see_val_loss.columns[1]],
             color='r', label='validation')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()

    # To load the display window
    plt.show()


def train(data):
    # Dividir os dados em dois segmentos: treino e validação numa relação de 80% para 20%
    train_data = data.sample(frac=0.8, random_state=42)
    val_data = data.drop(train_data.index)

    # Criar de fato os datasets e os dataloaders para treinamento e validação com uso das classes
    train_dataset = VoltageVelocityDataset(train_data)
    val_dataset = VoltageVelocityDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)

    # Define a rede MLP e a função de perda (loss function) como error mean squared
    mlp = MLP(input_dim=1, output_dim=1, hidden_dim=hidden_size,
              num_hidden_layers=hidden_layers)
    criterion = nn.MSELoss()

    # Definindo o uso de CPU ou GPU para processamento da rede
    mlp = mlp.to(device)

    # Define o tipo de ativação e o learning rate
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    # treino da rede MLP no data set todo

    # definição dos dataframes que guardarão os dados da evolução dos erros ao longo do treino
    see_train_loss = np.empty([1, 2]).astype(float)
    see_val_loss = np.empty([1, 2]).astype(float)
    idx_train = 0
    idx_val = 0

    for epoch in range(EPOCHS):
        train_loss = 0.0
        for X, Y in train_loader:
            # Feedforward
            outputs = mlp(X)
            loss = criterion(outputs, Y)

            # Backpropagation e ativação dos neurônios
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            see_train_loss = np.append(
                see_train_loss, [idx_train, loss.item()]).reshape(-1, 2)
            train_loss += loss.item() * X.shape[0]
            idx_train = idx_train+1

        # Cálculo, por rodada, do erro da validação
        with torch.no_grad():
            val_loss = 0.0
            for X, Y in val_loader:
                outputs = mlp(X)
                loss = criterion(outputs, Y)
                see_val_loss = np.append(
                    see_val_loss, [idx_val, loss.item()]).reshape(-1, 2)
                val_loss += loss.item() * X.shape[0]
                idx_val = idx_val+1

        # Monstrar o progresso
        if epoch % 100 == 0:
            print("| Epoch {:4} | train loss {:4.4f} | val loss {:4.4f} | ".format(
                epoch, train_loss / len(train_dataset), val_loss / len(val_dataset)))
    print('\n\nidx_train, idx_val: \n', idx_train, idx_val)
    return mlp, see_train_loss, see_val_loss


# Evaluate the MLP on the entire dataset
def predict(mlp, data):
    with torch.no_grad():
        X = torch.tensor(
            data[input_df_name].values).float().unsqueeze(1).to(device)
        Y = torch.tensor(
            data[output_df_name].values).float().unsqueeze(1).to(device)
        predictions = mlp(X)
        accuracy = ((predictions - Y) ** 2).mean().sqrt().item()
        print("Test accuracy| Mean Loss: {:.4f}".format(accuracy))
        return predictions, accuracy


def trained_info(data, predicted):
    df_data = data[output_df_name]
    diff = (predicted - df_data)
    diff_media = diff.abs().mean().to_numpy()[0]
    diff_max = diff.abs().max().to_numpy()[0]
    diff_min = diff.abs().min().to_numpy()[0]
    # print(f"df_data:\n{df_data}, predicted:\n{predicted}")
    # print( f'\nMédia da diferença: {diff_media:6.6f}\nMáxima diferença:   {diff_max:6.6f}\nMínima diferença:   {diff_min:6.6f}\n')
    return diff_media, diff_max, diff_min


def save_model(model):
    torch.save(model.state_dict(),
               f'{dir_base}/model_mlp_v{VERSION}_cluster.pth')


def runFake_df():
    mlp_saved = MLP(input_dim=1, output_dim=1,
                    hidden_dim=hidden_size, num_hidden_layers=hidden_layers)
    # mlp_saved = torch.load(f'{dir_base}/model_mlp_v{VERSION}_cluster.pth')
    df_full = pd.read_csv(
        "/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/pytorch_in/src/2_cluster_architecture/dadosFake_comAleatoriedade3M6.csv", sep=",")
    if (GRAPHS):
        predicted, accuracy = predict(mlp_saved, df_full)
        print(f"\n\tpredicted\n {predicted}")
        print(f"\n\taccuracy\n {accuracy}")
        # show_graphs(df, predicted)


def main():
    if (MODE):
        model, train_loss, validation_loss = train(df)
        predicted, accuracy = predict(model, df)
        predicted = pd.DataFrame(predicted)
        train_loss = pd.DataFrame(train_loss)
        validation_loss = pd.DataFrame(validation_loss)

        train_loss = train_loss.rename(columns={train_loss.columns[0]: 'time'})
        train_loss = train_loss.rename(
            columns={train_loss.columns[1]: 'error_train'})
        validation_loss = validation_loss.rename(
            columns={validation_loss.columns[0]: 'time'})
        validation_loss = validation_loss.rename(
            columns={validation_loss.columns[1]: 'error_val'})
        trained_info(df, predicted)

        if SAVE == True:
            save_model(model)

        if EXPORT_DATA == True:
            export_data(df, predicted, train_loss, validation_loss)
        # por ultimo, mostrar os graficos
        if GRAPHS == True:
            show_graphs(df, predicted, train_loss, validation_loss)

    else:
        runFake_df()
        pass


if __name__ == '__main__':
    main()
