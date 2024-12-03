# model-lstm

## Deploy com Docker

Para facilitar o deploy e garantir a portabilidade, o projeto inclui um Dockerfile para criar uma imagem Docker da aplicação.

### Passos:

1. Construir a imagem Docker:

    ```bash
    docker build -t stock-predictor .
    ```

2. Executar o contêiner:

    ```bash
    docker run -p 5000:5000 stock-predictor
    ```

A API estará disponível em [http://localhost:5000](http://localhost:5000).

**Nota:** Certifique-se de que o Docker está instalado e em execução no seu sistema.