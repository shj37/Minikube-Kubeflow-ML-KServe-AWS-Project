# ML Deployment with Kubeflow and KServe

![Project Overview](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*vkIKTzeZQ-497PrR7Lklzg.jpeg)

## Overview  
This project sets up a machine learning deployment using Kubeflow and KServe on Kubernetes. It handles the full ML lifecycle—data prep, training, storage, and inference—using Minikube locally and AWS for cloud support.

## Objectives  
- Automate ML workflows with Kubeflow Pipelines.  
- Store and version models in AWS S3.  
- Serve models with KServe for inference.  
- Keep the process scalable and repeatable.

## Setup Instructions  
1. **AWS Setup**:  
   - Start an EC2 instance (Ubuntu 24.04, t3.2xlarge).  
   - Open required ports in security groups.  
   - Create an S3 bucket for models.  
   - Set IAM roles for S3 access.  

2. **EC2 Prep**:  
   - Install Docker, Kubectl, Python, AWS CLI.  
   - Configure AWS credentials.  

3. **Minikube**:  
   - Install and launch Minikube with adequate resources.  

4. **Kubeflow**:  
   - Install Kubeflow Pipelines SDK.  
   - Deploy Kubeflow and open the UI.  

5. **Pipeline**:  
   - Define, compile, and run the ML pipeline in Kubeflow.  

6. **KServe**:  
   - Install KServe and connect it to S3.  
   - Deploy and test the model with sample data.  

For a full walkthrough, see the [article](https://medium.com/@jushijun/deploying-machine-learning-models-with-kubeflow-and-kserve-a-comprehensive-guide-2e3d1449dc54).

**All credit to iQuant for the original project design and code.**


## References
- YouTube iQuanthttps://www.youtube.com/watch?v=TQypOccQ3lc&t=8s&ab_channel=iQuant
- https://github.com/iQuantC/Kubeflow-KServe