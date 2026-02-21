FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies early for better caching
COPY requirement.txt ./
RUN pip install --no-cache-dir -r requirement.txt

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p data rag_store

EXPOSE 8000

CMD ["python", "main.py"]