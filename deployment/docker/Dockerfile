# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster AS python-base

# Install R dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python and R
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV R_HOME=/usr/lib/R

# Set working directory
WORKDIR /app

# Copy the requirements and R packages lists
COPY deployment/api/requirements.txt .
COPY deployment/api/packages.R /app/packages.R

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install R dependencies
RUN Rscript packages.R

# Copy the rest of the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8080

# Define entrypoint
CMD ["python", "deployment/api/app.py"]