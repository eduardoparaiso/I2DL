meu_projeto_deep_learning/
│
├── data/                    # Dados brutos e processados
│   ├── raw/                 # Dados brutos (originais, sem tratamento)
│   ├── processed/           # Dados já tratados e prontos para uso
│   └── external/            # Dados de fontes externas
│
├── notebooks/              # Notebooks para experimentação e EDA
│   ├── exploracao.ipynb    
│   └── testes_modelos.ipynb
│
├── src/                    # Código fonte do projeto
│   ├── __init__.py
│   ├── data/               # Scripts para carregamento e transformação de dados
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/           # Engenharia de atributos
│   │   └── extract_features.py
│   ├── models/             # Definição, treinamento e avaliação dos modelos
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── utils/              # Funções auxiliares
│   │   └── metrics.py
│   └── visualization/      # Scripts de visualização
│       └── plot_results.py
│
├── experiments/            # Resultados, logs e checkpoints de experimentos
│   ├── logs/
│   ├── runs/
│   └── checkpoints/
│
├── config/                 # Arquivos de configuração (YAML, JSON, etc.)
│   └── config.yaml
│
├── models/                 # Modelos treinados salvos
│   └── modelo_final.pth
│
├── scripts/                # Scripts para executar treinos, inferência etc.
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt        # Dependências do projeto
├── environment.yml         # Alternativa com Conda
├── README.md               # Descrição do projeto
├── .gitignore              # Arquivos a serem ignorados pelo Git
└── setup.py                # Script de instalação (se for um pacote)