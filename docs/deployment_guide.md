# Deployment Guide

## Overview

This guide outlines the deployment process for anomaly detection models in both Python and R environments. The models can be deployed on cloud platforms such as AWS and GCP using Docker.

### 1. Docker

- **Dockerfile**: A Dockerfile is provided to containerize the model along with its dependencies. It supports both Python and R environments.
- **docker-compose.yml**: If multiple containers are needed (e.g., for scaling), `docker-compose.yml` can be used to orchestrate them.

### 2. Cloud Deployment

- **AWS Deployment**: The `deploy_aws.py` script automates the deployment of the model on AWS.
- **GCP Deployment**: The `deploy_gcp.py` script handles deployment on Google Cloud.

### 3. API Deployment

The models can be served through a REST API using Flask (for Python) or Plumber (for R).

- **app.py**: Python API for model serving.
- **app.R**: R API for model serving.
