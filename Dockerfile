FROM python:3.9.18-slim-bullseye



COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgomp1 \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip \
    #&& pip install --no-cache-dir -r requirements.txt \
    && pip install -r requirements.txt \
    && rm requirements.txt

ENV AWS_DEFAULT_REGION=us-east-1



COPY model_pipeline/training/. .


CMD ["python","main_train.py"]