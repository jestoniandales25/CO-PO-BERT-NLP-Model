# Dockerfile
FROM python:3.10-slim

# avoid stdout buffering (helpful in logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install system deps for torch (minimal) and wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
 && rm -rf /var/lib/apt/lists/*

# copy
COPY requirements.txt .

# install python requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy app files
COPY . .

# Expose port (HF uses 7860 by convention)
EXPOSE 7860

# run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
