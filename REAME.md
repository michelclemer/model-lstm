# Previsão de Preços de Ações com LSTM

Este projeto cria um modelo preditivo usando LSTM para prever o valor de fechamento de uma ação específica e disponibiliza uma API para realizar previsões.

## Estrutura do Projeto

- `data_preprocessing.py`: Coleta e pré-processa os dados.
- `utils.py`: Funções auxiliares.
- `model_training.py`: Constrói, treina e salva o modelo.
- `model_prediction.py`: Carrega o modelo e realiza previsões.
- `app.py`: API Flask para servir o modelo.
- `requirements.txt`: Dependências do projeto.
- `Dockerfile`: Para construção da imagem Docker.
- `README.md`: Documentação do projeto.

## Instruções

### Pré-requisitos

- Python 3.9 ou superior
- Pip
- Docker (opcional, para containerização)

### Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
