import os
import time
import subprocess
from google.cloud import storage, logging_v2
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

# Constants
PROJECT_ID = "gcp-project-id"
REGION = "us-central1"
CLUSTER_NAME = "anomaly-detection-cluster"
DOCKER_IMAGE = "gcr.io/gcp-project-id/anomaly-detection:latest"
DEPLOYMENT_NAME = "anomaly-detection-deployment"
CONTAINER_PORT = 5000
GCS_BUCKET_NAME = "anomaly-detection-models"
MODEL_PATH = "models/model_file.pkl"
LOG_NAME = "anomaly-detection-logs"
HEALTH_CHECK_URL = "/health"
SERVICE_NAME = "anomaly-detection-service"
NAMESPACE = "default"

# Authenticate with Google Cloud
def authenticate_gcloud():
    """Authenticate GCP using default service account."""
    print("Authenticating Google Cloud...")
    credentials = GoogleCredentials.get_application_default()
    return credentials

# Create GKE Cluster
def create_gke_cluster():
    """Create a Google Kubernetes Engine (GKE) cluster."""
    print("Creating GKE cluster...")
    try:
        credentials = authenticate_gcloud()
        service = discovery.build('container', 'v1', credentials=credentials)
        cluster_config = {
            'name': CLUSTER_NAME,
            'initialNodeCount': 3,
            'nodeConfig': {
                'machineType': 'n1-standard-1',
                'diskSizeGb': 100,
                'oauthScopes': [
                    'https://www.googleapis.com/auth/devstorage.read_write',
                    'https://www.googleapis.com/auth/logging.write',
                    'https://www.googleapis.com/auth/cloud-platform'
                ]
            }
        }
        request = service.projects().zones().clusters().create(
            projectId=PROJECT_ID,
            zone=REGION,
            body=cluster_config
        )
        response = request.execute()
        print(f'Cluster creation in progress: {response["name"]}')
        wait_for_cluster_creation()
    except Exception as e:
        print(f"Error creating GKE cluster: {e}")

# Wait for GKE cluster to be ready
def wait_for_cluster_creation():
    """Wait for the GKE cluster to become ready."""
    print("Waiting for GKE cluster to be ready...")
    time.sleep(60)  # Simulate waiting, ideally, poll GKE for status.

# Deploy Docker Image to GKE
def deploy_to_gke():
    """Deploy Docker image to GKE cluster."""
    try:
        print("Deploying to GKE cluster...")
        subprocess.run([
            "kubectl", "create", "deployment", DEPLOYMENT_NAME,
            "--image", DOCKER_IMAGE
        ], check=True)
        subprocess.run([
            "kubectl", "expose", "deployment", DEPLOYMENT_NAME,
            "--type=LoadBalancer", f"--port={CONTAINER_PORT}", f"--target-port={CONTAINER_PORT}"
        ], check=True)
        check_deployment_status()
    except subprocess.CalledProcessError as e:
        print(f"Deployment error: {e}")

# Upload Model to Google Cloud Storage
def upload_model_to_gcs():
    """Upload model to Google Cloud Storage (GCS)."""
    try:
        print("Uploading model to GCS...")
        client = storage.Client()
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        blob.upload_from_filename(MODEL_PATH)
        print(f'Model uploaded successfully: gs://{GCS_BUCKET_NAME}/{MODEL_PATH}')
    except Exception as e:
        print(f"Error uploading model: {e}")

# Set up Health Check for GKE Service
def setup_health_check():
    """Set up health check for deployed service."""
    print(f"Setting up health check for service {SERVICE_NAME}...")
    try:
        subprocess.run([
            "kubectl", "create", "service", "http", SERVICE_NAME,
            f"--tcp={CONTAINER_PORT}:{CONTAINER_PORT}"
        ], check=True)
        subprocess.run([
            "kubectl", "apply", "-f", "-",
            "--namespace", NAMESPACE
        ], input=f"""
        apiVersion: v1
        kind: Service
        metadata:
          name: {SERVICE_NAME}
          namespace: {NAMESPACE}
        spec:
          ports:
          - port: {CONTAINER_PORT}
            targetPort: {CONTAINER_PORT}
          selector:
            app: {DEPLOYMENT_NAME}
          type: LoadBalancer
        """.encode('utf-8'))
        print("Health check setup complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up health check: {e}")

# Check Deployment Status
def check_deployment_status():
    """Check the status of the deployment."""
    print("Checking deployment status...")
    try:
        output = subprocess.run(["kubectl", "get", "deployments"], capture_output=True, text=True)
        print(f"Deployment status:\n{output.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error checking deployment status: {e}")

# Setup Logging in Google Cloud
def setup_logging():
    """Set up Google Cloud Logging."""
    try:
        print("Setting up Google Cloud Logging...")
        client = logging_v2.Client()
        logger = client.logger(LOG_NAME)
        logger.log_text("Anomaly detection model deployment started.")
        print("Logging setup complete.")
    except Exception as e:
        print(f"Error setting up logging: {e}")

# Monitor Logs in Real-time
def monitor_logs():
    """Monitor logs from GKE deployment."""
    print("Monitoring logs in real-time...")
    try:
        subprocess.run([
            "kubectl", "logs", "-l", f"app={DEPLOYMENT_NAME}", "-f"
        ])
    except subprocess.CalledProcessError as e:
        print(f"Error monitoring logs: {e}")

# Rollback Deployment
def rollback_deployment():
    """Rollback the deployment in case of errors."""
    print("Rolling back deployment...")
    try:
        subprocess.run([
            "kubectl", "rollout", "undo", f"deployment/{DEPLOYMENT_NAME}"
        ], check=True)
        print(f"Rollback completed for {DEPLOYMENT_NAME}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during rollback: {e}")

# Main deployment process
if __name__ == "__main__":
    try:
        print("Starting GCP deployment process...")
        create_gke_cluster()
        upload_model_to_gcs()
        deploy_to_gke()
        setup_health_check()
        setup_logging()
        monitor_logs()
        print("Deployment successful.")
    except Exception as e:
        print(f"Deployment failed: {e}")
        rollback_deployment()