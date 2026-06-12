FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    confluent-kafka \
    psycopg2-binary \
    pydantic \
    numpy \
    scikit-surprise

COPY . /app/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
