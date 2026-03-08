FROM python:3.11-slim

WORKDIR /app

# Install system deps for numpy/sklearn compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI service
CMD ["uvicorn", "part4_api:app", "--host", "0.0.0.0", "--port", "8000"]
