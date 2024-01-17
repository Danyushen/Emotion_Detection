FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY config.yaml config.yaml
COPY web.py web.py

# pull model from dvc and copy to checkpoints
# RUN dvc pull
# COPY models/checkpoints/ models/checkpoints/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir

EXPOSE 8000

ENTRYPOINT ["python", "-u", "src/web/web.py"]
