# IOT sensor data monitoring

## Project overview

In this project we used the Gradient Boosting Classifier to predict failures in IoT sensor data monitoring. By analyzing sensor readings such as population (popul), temperature (temp), pressures (outpressure, inpressure), and various parameters like atemp, selfLR, ClinLR, DoleLR, and PID, the model learns patterns that are indicative of system failures. The target variable, fail, denotes whether a failure has occurred. Gradient Boosting is employed to build a robust predictive model, providing insights into potential failures in real-time, enabling early detection and proactive maintenance for IoT systems.The dataset used in this project is available here : `https://github.com/IBM/iot-predictive-analytics/blob/master/data/iot_sensor_dataset.csv`

## Setup and Installation

To set up the environment and install dependencies, follow the steps below:

- Option 1: using `requirements.txt`
```
pip install -r requirements.txt
```
- Option 2: using `setup.py`

Alternatively, if you'd like to install the project as a package : 
```
pip install .
```

## Directory structure
The project has the following directory structure:
```
.
├── config/                     
├── data/                       
├── deployment/                 
├── logs/                       
├── reports/                    
├── results/                   
├── scripts/                    
├── src/                        
├── tests/                     
├── Dockerfile                 
├── docker-compose.yaml         
└── requirements.txt  

```

Key Directories :

- `config/`: code for setting up paths for loading and saving data.
- `data/`: stores raw, processed, and inference data files, including the churn dataset.
- `deployment/`: includes Docker image and MLflow configurations for model deployment.
- `logs/`: logs generated during experimentation and API prediction.
- `reports/` :  EDA plots.
- `scripts/`: scripts for training, inference, and running the entire pipeline.
- `results/`: stores trained models, evaluation results, and scaler objects.
- `src/`:  core source code.
- `tests/`: unit tests for the core modules.


## Project usage
1. Running the Full Pipeline
 
To run the entire churn prediction pipeline, execute the main script located at `/scripts/run_all_scripts.py`. This script will:

- Process raw data from `/data/raw/` 
- Train the churn prediction model
- Evaluate the model and save the results to `/results/`
- Perform inference on sample data located in `/data/inference/`

Alternatively, you can run the individual steps of the pipeline:
- Training: run `/scripts/training_script.py` to train the model.
- Inference: run `/scripts/inference_script.py` to make predictions on new data.

2. Running the API for prediction 

To serve the model via an API, execute the bash script `/src/serving/fast_api/start_fastapi.sh` to start the FastAPI server. You can also test the API by running the `test_api.sh` script.

3. Running the API as a Docker Container 

To run the API in a Docker container:
- Build the Docker image

`docker build -t sensor-data-monitoring .`

- Run the Docker container

`docker run -d -p 8000:8000 sensor-data-monitoring`

If you'd like to make frequent changes to the project files while the container is running, you can use Docker Compose to manage the container:

`docker-compose up --build`


