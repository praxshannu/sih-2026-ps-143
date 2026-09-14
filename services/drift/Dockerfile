FROM python:3.11-slim AS base
WORKDIR /app
RUN groupadd -r sentinel && useradd -r -g sentinel sentinel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
RUN chown -R sentinel:sentinel /app
USER sentinel
EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
