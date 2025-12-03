FROM apache/spark:3.5.2-python3

USER root
WORKDIR /app


COPY requirements.txt . 
COPY app/requirements.txt ./app-requirements.txt
# Install root + app requirements and add scikit-learn and joblib for sklearn-based training
RUN pip install --no-cache-dir -r requirements.txt -r app-requirements.txt scikit-learn joblib

COPY ./app /app

CMD ["spark-submit", "main.py"]
