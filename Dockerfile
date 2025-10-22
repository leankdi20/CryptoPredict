FROM python:3.12-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos
COPY . /app

# Instala las dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Comando por defecto
CMD ["python", "main.py", "--loop"]
