# **ML FastAPI Docker Deployment Project**

This project demonstrates an end-to-end pipeline to create, deploy, and serve a machine learning model using **FastAPI**, **Docker**, and **AWS Elastic Container Registry (ECR)**. It covers training the ML model, serving it with FastAPI, containerizing it with Docker, and deploying it to AWS.

---

## **Table of Contents**
1. [Requirements](#requirements)
2. [Setup Instructions](#setup-instructions)
3. [Project Structure](#project-structure)
4. [Step-by-Step Guide](#step-by-step-guide)
    - [Train and Save the Model](#train-and-save-the-model)
    - [Create a FastAPI Application](#create-a-fastapi-application)
    - [Test the Application Locally](#test-the-application-locally)
    - [Containerize with Docker](#containerize-with-docker)
    - [Push Docker Image to AWS ECR](#push-docker-image-to-aws-ecr)
    - [Deploy the Application](#deploy-the-application)
5. [Test the Deployed Application](#test-the-deployed-application)
6. [Future Improvements](#future-improvements)

---

## **Requirements**

### Tools to Install
1. **Python** (>=3.8)
   - Install from [https://www.python.org/](https://www.python.org/)
2. **Docker**
   - Install from [https://www.docker.com/](https://www.docker.com/)
3. **AWS CLI**
   - Install from [https://aws.amazon.com/cli/](https://aws.amazon.com/cli/)
4. **Git**
   - Install from [https://git-scm.com/](https://git-scm.com/)

### Python Libraries
Install the following libraries:
```bash
pip install fastapi uvicorn scikit-learn joblib
```
---

## **Setup Instructions**

### Clone the Repository
```bash
git clone https://github.com/your-repository/ml-fastapi-docker.git
cd ml-fastapi-docker
```
ml-fastapi-docker/ ├── app/ │ ├── init.py # Initialization file for the app module │ ├── main.py # FastAPI application code │ ├── model.pkl # Trained machine learning model file ├── create_model.py # Script to train and save the ML model ├── Dockerfile # Dockerfile to containerize the application ├── requirements.txt # Python dependencies for the project ├── .dockerignore # Files and folders to ignore during Docker build └── README.md # Detailed instructions and documentation
