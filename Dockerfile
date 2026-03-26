FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Vars configuráveis em runtime
ENV PORT=8000
ENV MODEL_PATH=artefatos/pacote_pipeline.json
ENV PACIENCIA=20

COPY src/ /app/src/
COPY serve.py /app/serve.py

RUN pip install --no-cache-dir "numpy>=1.24"

EXPOSE ${PORT}

CMD ["python", "serve.py"]

