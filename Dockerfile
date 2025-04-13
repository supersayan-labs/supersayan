FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Julia
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN curl -LO https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz && \
    tar -xzf julia-1.8.5-linux-x86_64.tar.gz && \
    rm julia-1.8.5-linux-x86_64.tar.gz && \
    mv julia-1.8.5 /opt/julia && \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia

# Set up directories
RUN mkdir -p /app/src /app/scripts /app/models

# Copy the project files
COPY pyproject.toml /app/
COPY README.md /app/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY docs/ /app/docs/

# Install Python dependencies and the package
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install -e .

# Install additional dependencies for the server
RUN uv pip install fastapi uvicorn requests

# Create directory for models
RUN mkdir -p /tmp/supersayan/models

# Set environment variables
ENV PYTHONPATH=/app
ENV MODELS_DIR=/tmp/supersayan/models

# Initialize Julia dependencies (pre-compile)
RUN python -c "from supersayan.core.bindings import initialize_julia; initialize_julia()"

# Expose the API port
EXPOSE 8000

# Run the server
CMD ["python", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8000"]