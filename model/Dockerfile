FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    scikit-surprise \
    pandas \
    numpy \
    psycopg2-binary \
    confluent-kafka

COPY . /app/

CMD ["python", "-m", "model.train"]
