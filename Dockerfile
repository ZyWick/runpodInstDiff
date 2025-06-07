# Use Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git and any other required tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy your application files
COPY handler.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
CMD ["python", "handler.py"]
