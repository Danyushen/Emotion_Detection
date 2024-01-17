# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY config.yaml config.yaml

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir
#RUN python -u src/data/make_dataset.py


ENTRYPOINT ["python", "-u", "src/train_model.py"]
