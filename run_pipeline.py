import mlflow
import os
import subprocess
import numpy as np


def run_command(command, cwd):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise Exception(f"Command failed with return code {result.returncode}")
    
def startup():
    run_command("poetry install", cwd=os.path.abspath("../rb_model"))
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

def run_data_ingestion():
    run_command("poetry run python ingest.py", cwd=os.path.abspath("../rb_model/data_ingestion"))

def run_model_training():
    run_command("poetry run python train.py", cwd=os.path.abspath("../rb_model/model_training"))

def run_model_evaluation():
    run_command("poetry run python evaluate.py", cwd=os.path.abspath("../rb_model/model_evaluation"))

if __name__ == "__main__":
    
    # set seed for reproducibility
    np.random.seed(123)


    # Set the MLflow tracking URI to the local server
    startup()
    run_data_ingestion()
    run_model_training()
    run_model_evaluation()