FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY src/ /app/src/
COPY serve.py /app/serve.py

RUN pip install --no-cache-dir numpy

EXPOSE 8000

CMD ["python", "serve.py"]

