FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpng-dev \
        libfreetype6-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
