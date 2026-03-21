# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (sometimes needed for building certain python packages)
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . /app/backend

# Set the python path so the backend module can be found
ENV PYTHONPATH=/app

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]