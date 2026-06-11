FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    confluent-kafka \
    psycopg2-binary \
    pydantic \
    numpy

COPY . /app/

CMD ["python", "-m", "pipeline.consumer"]
