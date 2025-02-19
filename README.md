# API de Previsão de Preço de Ações

## Visão Geral

Este projeto implementa uma API REST utilizando FastAPI para prever preços de ações utilizando um modelo de rede neural Long Short-Term Memory (LSTM). A API permite que os usuários solicitem previsões para um símbolo de ação específico e intervalo de datas.

## Estrutura do Projeto

```
stock-price-forecaster/
├── data/
│ ├── raw/ # Dados brutos baixados de fontes (inicialmente vazios)
│ └── processed/ # Dados processados (inicialmente vazios)
├── logs/ # Arquivos de log da aplicação
├── models/ # Modelos salvos (modelo LSTM e Scaler)
│ ├── Scaler_model.pkl
│ └── best_lstm_model.keras
├── notebooks/ # Notebooks Jupyter para experimentação e exploração do pipeline
│ └── pipeline.ipynb
├── src/ # Código fonte da aplicação
│ ├── data_handler.py # Funções para baixar e pré-processar os dados das ações
│ ├── logger.py # Configuração de log
│ ├── lstm_model.py # Funções para criar, treinar e salvar o modelo LSTM
│ ├── model_building.py # Script para construir e treinar o modelo LSTM
│ └── model_predict.py # Funções para carregar modelo, scaler e fazer previsões
├── routes/ # Definições de rotas FastAPI
│ ├── routes.py # Rotas de endpoints da API (ex: /predict)
├── app.py # Inicialização e início da aplicação FastAPI
├── Dockerfile # Dockerfile para containerizar a aplicação
├── docs/ # Documentação do projeto (inicialmente contém este README)
│ └── README.md
├── requirements.txt # Dependências Python
└── .gitignore # Arquivo gitignore
```

**Descrição das Principais Pastas e Arquivos:**

*   **`data/`**: Contém dados brutos e processados das ações. Atualmente, estas pastas estão previstas para gerenciamento de dados no futuro, mas não são ativamente usadas no código fornecido, pois os dados são baixados dinamicamente.
*   **`logs/`**: Armazena arquivos de log gerados pela aplicação para depuração e monitoramento.
*   **`models/`**: Contém os modelos de aprendizado de máquina treinados:
    *   `Scaler_model.pkl`: O modelo `MinMaxScaler` utilizado para normalização dos dados.
    *   `best_lstm_model.keras`: O modelo LSTM treinado no formato Keras, salvo por `ModelCheckpoint` durante o treinamento.
*   **`notebooks/pipeline.ipynb`**: Um Jupyter Notebook demonstrando o pipeline de dados, treinamento de modelo e exploração.
*   **`src/`**: Contém o código fonte principal em Python:
    *   `data_handler.py`: Responsável por baixar os dados das ações do Yahoo Finance e Alpha Vantage, padronização de dados e pré-processamento para o modelo LSTM.
    *   `logger.py`: Configura o sistema de logging para a aplicação.
    *   `lstm_model.py`: Define funções para criar, compilar, treinar e salvar o modelo LSTM. Inclui uma métrica personalizada RMSE e callbacks como `EarlyStopping` e `ModelCheckpoint`.
    *   `model_building.py`: Orquestra todo o pipeline de construção do modelo, desde o download dos dados até o salvamento do modelo. Executável como script para treinar um novo modelo.
    *   `model_predict.py`: Contém funções para carregar o modelo e scaler treinados e realizar previsões de preços de ações. Otimizado para uso na API FastAPI.
*   **`routes/routes.py`**: Define as rotas da API utilizando o `APIRouter` do FastAPI. Atualmente inclui o endpoint `/predict` para previsão de preços de ações.
*   **`app.py`**: Inicializa a aplicação FastAPI, configura o log, carrega o modelo e scaler treinados na inicialização e inclui as rotas da API definidas em `routes/routes.py`.
*   **`Dockerfile`**: Contém as instruções para construir uma imagem Docker e facilitar o deployment e containerização da API.
*   **`requirements.txt`**: Lista as dependências de pacotes Python necessárias para rodar o projeto.

## Como Começar

Siga os passos abaixo para configurar e rodar o projeto localmente.

### Pré-requisitos

*   **Python 3.9 ou superior**: Certifique-se de que o Python esteja instalado em seu sistema. Você pode baixá-lo em [python.org](https://www.python.org/).
*   **pip**: Instalador de pacotes Python (geralmente já vem com a instalação do Python).
*   **Ambiente Virtual (recomendado)**: É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto.
*   **Docker (opcional)**: Necessário se você quiser rodar a aplicação utilizando Docker. Você pode baixá-lo em [docker.com](https://www.docker.com/).
*   **Chave da API Alpha Vantage (opcional)**: Necessária se o Yahoo Finance não fornecer dados e a aplicação fizer fallback para o Alpha Vantage. Obtenha uma chave de API gratuita em [alphavantage.co](https://www.alphavantage.co/) e defina-a como uma variável de ambiente `ALPHA_KEY`.

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <repository_url>
    cd stock-price-forecaster
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv-tech  # Ou qualquer nome de sua preferência
    source venv-tech/bin/activate  # No Linux/macOS
    venv-tech\Scripts\activate  # No Windows
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Rodando a Aplicação

#### 1. Construir o Modelo LSTM (se ainda não o fez)

Antes de rodar a API, você precisa garantir que o modelo LSTM e o scaler estejam treinados e salvos no diretório `models/`. Você pode fazer isso rodando o script `model_building.py`:

```bash
python src/model_building.py --symbol MSFT
```

Você pode alterar o argumento `--symbol` para treinar um modelo para uma ação diferente. Após execução bem-sucedida, você encontrará o arquivo `Scaler_model.pkl` e `best_lstm_model.keras` no diretório `models/`.

#### 2. Iniciar a API FastAPI

Para rodar o servidor da API:

```bash
python app.py
```

Isso iniciará o servidor Uvicorn hospedando a aplicação FastAPI em http://0.0.0.0:8000. Você deverá ver logs indicando que o modelo e o scaler foram carregados e a API está pronta.

#### 3. Acessar a Documentação da API

O FastAPI gera automaticamente documentação interativa da API. Você pode acessá-la no seu navegador em:

* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

Use essas interfaces para explorar o endpoint `/predict` e testar as requisições da API.

### Uso - Endpoint de Previsão (/predict)

A API oferece um único endpoint para previsão de preço de ações:

Endpoint: `GET /predict`

Parâmetros de Requisição (Parâmetros de Query):

*   `symbol` (string, obrigatório): O símbolo da ação para previsão (ex: AAPL, MSFT, PETR4.SA).
*   `start_date` (string, obrigatório): A data de início para obter os dados históricos (formato: YYYY-MM-DD).
*   `end_date` (string, obrigatório): A data final para obter os dados históricos (formato: YYYY-MM-DD).

Exemplo de Requisição (usando curl):

```bash
curl "http://localhost:8000/predict?symbol=AAPL&start_date=2024-01-01&end_date=2024-02-01"
```

Exemplo de Requisição (usando Python requests):

```python
import requests

url = "http://localhost:8000/predict"
params = {
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-02-01"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    prediction_data = response.json()
    predicted_price = prediction_data['predicted_price']
    print(f"Preço previsto para AAPL: {predicted_price:.2f}")
else:
    print(f"Erro na requisição: {response.status_code}")
    print(response.json())
```

Formato de Resposta (JSON):

Em caso de previsão bem-sucedida (status HTTP 200 OK), a API retorna uma resposta JSON no seguinte formato:

```json
{
  "predicted_price": 175.50
}
```

Respostas de Erro:

Em caso de erros (por exemplo, parâmetros inválidos, dados não encontrados, erros de servidor), a API retornará respostas de erro HTTP (códigos de status como 400, 500) com um corpo JSON contendo detalhes do erro.

### Dockerização

O projeto inclui um Dockerfile para construir uma imagem Docker e facilitar o deployment.

Construa a imagem Docker:

```bash
docker build -t stock-price-forecaster-api .
```

Execute o contêiner Docker:

```bash
docker run -p 8000:8000 stock-price-forecaster-api
```

A API estará acessível em http://localhost:8000 na sua máquina local.

## Melhorias e Trabalhos Futuros

* Expandir o Conjunto de Funcionalidades: Incorporar mais recursos além do preço de fechamento, como volume, preço de abertura, máximo, mínimo e indicadores técnicos para melhorar a acurácia da previsão.
* Otimização de Hiperparâmetros: Otimizar sistematicamente os hiperparâmetros do modelo LSTM (número de camadas, unidades, taxa de dropout, otimizador, etc.) utilizando técnicas como Grid Search, Random Search ou Otimização Bayesiana.
* Métricas de Avaliação do Modelo: Implementar métricas de avaliação do modelo mais abrangentes e estratégias de validação, incluindo validação cruzada de séries temporais.
* Armazenamento e Gerenciamento de Dados: Integrar um banco de dados ou solução de armazenamento adequado para gerenciar dados históricos de ações, ao invés de depender exclusivamente do download de dados online.
* Autenticação e Segurança da API: Adicionar mecanismos de autenticação e autorização para proteger os endpoints da API.
* Melhoria no Tratamento de Erros e Logging: Melhorar o tratamento de erros e fornecer logs mais detalhados para facilitar a depuração e o monitoramento em produção.
* Pipeline CI/CD: Configurar um pipeline de Integração Contínua/Desdobramento Contínuo (CI/CD) para construção, teste e deployment automatizados da API.

## Licença

Apache License

## Autor

Ernane Domingues