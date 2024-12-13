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
# Project Directory Structure

ml-fastapi-docker/ ├── app/ │ ├── init.py # Initialization file for the app module │ ├── main.py # FastAPI application code │ ├── model.pkl # Trained machine learning model file ├── create_model.py # Script to train and save the ML model ├── Dockerfile # Dockerfile to containerize the application ├── requirements.txt # Python dependencies for the project ├── .dockerignore # Files and folders to ignore during Docker build └── README.md # Detailed instructions and documentation


## **Step-by-Step Guide**

### **1. Train and Save the Model**
1. **Create the ML Model**:
   - Open the file `create_model.py` and add:
     ```python
     import joblib
     from sklearn.datasets import load_iris
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.model_selection import train_test_split

     # Load dataset
     data = load_iris()
     X, y = data.data, data.target

     # Split dataset
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Train a RandomForest model
     model = RandomForestClassifier()
     model.fit(X_train, y_train)

     # Save the model
     joblib.dump(model, "app/model.pkl")
     print("Model saved as app/model.pkl")
     ```

2. **Run the Script**:
   ```bash
   python create_model.py
    ```
### **2. Create a FastAPI Application**

1. **Create `app/main.py`:**

   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   import joblib

   # Load the model
   model = joblib.load("app/model.pkl")

   # Initialize FastAPI app
   app = FastAPI()

   # Input schema
   class IrisInput(BaseModel):
       sepal_length: float
       sepal_width: float
       petal_length: float
       petal_width: float

   @app.get("/")
   def read_root():
       return {"message": "Welcome to the ML API"}

   @app.post("/predict/")
   def predict(input_data: IrisInput):
       features = [[
           input_data.sepal_length,
           input_data.sepal_width,
           input_data.petal_length,
           input_data.petal_width
       ]]
       prediction = model.predict(features)
       return {"prediction": int(prediction[0])}
   ```
### **3. Test the Application Locally**

Run the application locally:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
