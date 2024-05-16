# Gender Detection Project

This project aims to build a machine learning model for gender detection using Apache Spark for data processing, Apache Airflow for pipeline orchestration, MLflow for experiment tracking, and a REST API for model serving. The project also includes Prometheus for metrics instrumentation and Grafana for visualization.

## Contents

- Apache Spark
- Apache Airflow
- Python
- MLflow
- Flask
- Prometheus
- Grafana
- Docker

## Data Pipeline

The data pipeline is orchestrated using Apache Airflow. The `airflow/dags/data_pipeline.py` file contains the DAG definition, which consists of the following tasks:

1. **Data Acquisition**: Automates the process of retrieving raw data from various sources.
2. **Data Cleansing**: Handles missing values, removes duplicates, and performs other data cleaning operations.
3. **Data Transformation**: Applies necessary transformations to the data, such as feature engineering.
4. **Vectorization**: Converts the transformed data into a format suitable for machine learning models.

These tasks are implemented in the corresponding Python scripts under the `airflow/scripts` directory.

## Model Building and Training

The model building and training process is handled separately from the data pipeline. You can find the relevant code in the `notebooks/exploratory_analysis.ipynb` Jupyter Notebook, which explores the data, trains the model, and evaluates its performance.

## REST API

The `api/app.py` file contains a Flask application that exposes a REST API for serving the trained gender detection model. The API endpoints are:

- `/predict` (POST): Accepts input data and returns the predicted gender.

The `api/utils.py` file contains helper functions for loading the trained model and performing predictions.

## Metrics and Monitoring

Prometheus is used for metrics instrumentation, and the configuration is defined in the `prometheus/prometheus.yml` file. The REST API functions are instrumented to capture relevant metrics.

Grafana is used for visualizing the captured metrics. The `grafana/provisioning/dashboards` directory contains the Grafana dashboard configuration files, and the `grafana/provisioning/datasources` directory contains the data source configuration files.

## Deployment

The project can be deployed using Docker. The `Dockerfile` defines the Docker image for the REST API, and the `docker-compose.yml` file defines the services for the REST API, Prometheus, and Grafana.

To deploy the project, follow these steps:

1. Build the Docker image: `docker build -t gender-detection-api .`
2. Start the services: `docker-compose up -d`

The REST API will be accessible at `http://localhost:5000`, Prometheus at `http://localhost:9090`, and Grafana at `http://localhost:3000`.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b my-feature-branch`
3. Make your changes and commit them: `git commit -am 'Add some feature'`
4. Push your changes to your forked repository: `git push origin my-feature-branch`
5. Create a new pull request.
