# Étape 1 - Builder
FROM python:3.10 as builder

WORKDIR /app
COPY api/requirements.txt .
RUN pip install --user -r requirements.txt

# Étape 2 - Runtime
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY api/ .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 5000
CMD ["python", "app.py"]