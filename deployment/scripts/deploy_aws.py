import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.s3 import S3Uploader
from botocore.exceptions import ClientError
import logging
import os
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
aws_region = 'us-west-2'
bucket_name = 's3-bucket-name'
s3_model_path = 'models/anomaly_detection_model.tar.gz'
model_artifact = f's3://{bucket_name}/{s3_model_path}'
role = get_execution_role()
sess = sagemaker.Session()

# Function to check S3 bucket existence
def check_s3_bucket(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    if bucket.creation_date:
        logger.info(f'S3 bucket {bucket_name} exists.')
    else:
        raise Exception(f'S3 bucket {bucket_name} does not exist or cannot be accessed.')

# Upload the trained model artifact to S3
def upload_model_to_s3(local_model_path, s3_model_path):
    try:
        if os.path.exists(local_model_path):
            S3Uploader.upload(local_model_path, s3_model_path)
            logger.info(f'Model uploaded to S3: {s3_model_path}')
        else:
            raise FileNotFoundError(f"Model path {local_model_path} does not exist.")
    except ClientError as e:
        logger.error(f"Failed to upload model: {e}")
        raise

# Deploy the model using SageMaker
def deploy_model(model_artifact, instance_type='ml.m5.large', endpoint_name='anomaly-detection-endpoint'):
    try:
        container_image = sagemaker.image_uris.retrieve('pytorch', aws_region, version='1.9')
        
        model = sagemaker.model.Model(
            image_uri=container_image,
            model_data=model_artifact,
            role=role,
            sagemaker_session=sess
        )
        
        logger.info('Deploying the model...')
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        
        logger.info(f'Model deployed to endpoint: {endpoint_name}')
    except ClientError as e:
        logger.error(f"Error deploying model: {e}")
        raise

# Invoke SageMaker endpoint
def invoke_endpoint(endpoint_name, input_data):
    client = boto3.client('sagemaker-runtime', region_name=aws_region)
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=input_data
        )
        result = response['Body'].read().decode()
        logger.info(f'Inference result: {result}')
        return result
    except ClientError as e:
        logger.error(f"Error invoking endpoint: {e}")
        raise

# Clean up SageMaker resources (delete endpoint)
def clean_up_resources(endpoint_name):
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f'Endpoint {endpoint_name} deleted successfully.')
    except ClientError as e:
        logger.error(f"Failed to delete endpoint: {e}")
        raise

# CloudWatch Logging for Monitoring Model Inference
def setup_cloudwatch_logging(endpoint_name):
    logs_client = boto3.client('logs', region_name=aws_region)
    log_group_name = f'/aws/sagemaker/Endpoints/{endpoint_name}'

    try:
        # Check if log group exists, if not create it
        logs_client.create_log_group(logGroupName=log_group_name)
        logger.info(f'Log group created: {log_group_name}')
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
            logger.info(f'Log group already exists: {log_group_name}')
        else:
            logger.error(f"Error setting up CloudWatch logging: {e}")
            raise

# Track model performance in CloudWatch Metrics
def track_model_performance(endpoint_name):
    cloudwatch_client = boto3.client('cloudwatch', region_name=aws_region)

    # CloudWatch metrics logic
    cloudwatch_client.put_metric_data(
        Namespace='AnomalyDetectionModel',
        MetricData=[
            {
                'MetricName': 'InferenceLatency',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}],
                'Unit': 'Milliseconds',
                'Value': 150.0
            },
        ]
    )
    logger.info(f'Performance metric sent to CloudWatch for {endpoint_name}.')

# Main execution
if __name__ == "__main__":
    local_model_path = './anomaly_detection_model.tar.gz'
    endpoint_name = 'anomaly-detection-endpoint'
    
    # Step 1: Check if the S3 bucket exists
    check_s3_bucket(bucket_name)
    
    # Step 2: Upload the model to S3
    upload_model_to_s3(local_model_path, f's3://{bucket_name}/{s3_model_path}')
    
    # Step 3: Deploy the model to SageMaker
    deploy_model(model_artifact, instance_type='ml.m5.large', endpoint_name=endpoint_name)
    
    # Step 4: Setup CloudWatch logging
    setup_cloudwatch_logging(endpoint_name)
    
    # Step 5: Track performance metrics on CloudWatch
    track_model_performance(endpoint_name)
    
    # Step 6: Test inference with sample data
    test_data = json.dumps({"input_data": [0.1, 0.2, 0.3, 0.4]})
    result = invoke_endpoint(endpoint_name, test_data)
    logger.info(f"Test inference result: {result}")
    
    # lean up SageMaker resources after testing
    time.sleep(60)  # Wait for some time before cleanup to ensure all logs are written
    clean_up_resources(endpoint_name)