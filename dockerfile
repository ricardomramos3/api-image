# Use a imagem oficial do Python como base
FROM python:3.9

# Defina o diretório de trabalho como /app
WORKDIR /app

# Copie os arquivos de código-fonte para o contêiner
COPY worker.py .
COPY index.html .

# Instale as dependências do seu projeto
RUN pip install fastapi==0.68.1 uvicorn==0.15.0 pydantic diffusers==0.21.0 transformers==4.22.0 torch==1.10.0 Pillow==8.4.0

# Exponha a porta 8000
EXPOSE 8000

# Comando de execução para iniciar o servidor FastAPI
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
