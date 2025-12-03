FROM apache/spark:3.5.2-python3

USER root
WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

CMD ["spark-submit", "main.py"]
