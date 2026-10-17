FROM python:3.11-slim

# OpenCV runtime libraries, and a compiler for insightface's native extension
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY helmet_monitor ./helmet_monitor
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "helmet_monitor.main:app", "--host", "0.0.0.0", "--port", "8000"]
