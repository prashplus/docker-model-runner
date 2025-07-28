# Dockerfile for Model Runner with Llama 3.2 Support
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for transformers
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

# Create cache directory
RUN mkdir -p /app/cache

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash modeluser
RUN chown -R modeluser:modeluser /app
USER modeluser

# Expose port for the model server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the model server
CMD ["python", "src/server.py"]
