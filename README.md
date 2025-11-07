## Purpose
This is a sample code repository that demonstrates how to organize code for an ML business solution. This code repository is created as part of creating a Project in SageMaker. It's intended to showcase an understanding of MLOps concepts and an understanding of AWS SageMaker. It mostly follows the [AWS Blogs MLOps Guide](https://github.com/aws-samples/mlops-sagemaker-github-actions).

In particular, the intention of this project is to acquire and demonstrate end-to-end MLOps skills with AWS SageMaker. I set up a pipeline on AWS SageMaker (from data to model registry), and did CI/CD with GitHub Actions to build and deploy the model from staging to prod.

### What this showcases
- **SageMaker Pipelines:** process/train → evaluate → register
- **Model Registry:** versioned model referenced by deploy workflow
- **CI/CD with GitHub Actions:** build/update pipeline; deploy to staging → production
- **SageMaker Endpoints:** serve staging and production endpoints

## Architecture Overview
![Amazon SageMaker and GitHub Actions Architecture](/img/pipeline.png)

Each commit with changes in the `pipelines/` folder triggers the GitHub workflow `BuildSageMakerModel` defined in `.github/workflows/build.yml`. This workflow is triggering an AWS SageMaker pipeline to preprocess the data, train the model, evaluate it and then register the model within the model registry if the model performed well enough. From there, the model needs a manual approval through SageMaker. After the manual approval, the GitHub workflow `DeploySageMakerModel` consisting of 2 jobs needs to be triggered manually to prepare the model for the staging and production environments. After being successfully deployed to staging through the first job, the deployment from staging to production (2nd job) needs an additional review through GitHub.

## Results & Evidence

1. Successful SageMaker pipeline execution

    ![Amazon SageMaker and GitHub Actions Architecture](/img/successful-sagemaker-pipeline-execution.png)

    Note: The training step is a "Process data" step because the AWS free tier doesn't provide training quota. It's circumventing that by making the training step a process data step.

2. Model Registry

    ![Amazon SageMaker and GitHub Actions Architecture](/img/model-registry.png)

3. GitHub CI/CD

    ![Amazon SageMaker and GitHub Actions Architecture](/img/github-cicd.png)

4. Endpoints in service

    ![Amazon SageMaker and GitHub Actions Architecture](/img/sagemaker-endpoints-running.png)

5. Inference through production Endpoint

    Run the production endpoint from a Terminal. AWS CLI needs to be installed if run locally.
    Replace `type` with `cat` to run on bash instead of Windows. Make sure it runs on the correct region, otherwise also add the `--region <your-region>` parameter.

    ```
    aws sagemaker-runtime invoke-endpoint --endpoint-name aws-sagemaker-mlops-pipeline-prod --content-type text/csv --cli-binary-format raw-in-base64-out --body "0.383148412,0.273297349,-0.227545026,-0.153451984,-0.046713787,-0.046474046,-0.322093154,0,0,1" out.json && type out.json
    ```

    This will return something like:
    ```
    {
        "ContentType": "text/csv; charset=utf-8",
        "InvokedProductionVariant": "AllTraffic"
    }

    9.27243709564209
    ```
    Note: This is a real sample, the correct number of rings of this abalone is 10.

## Project and Model Description
In this example, the abalone age prediction problem using the abalone dataset (see below for more on the dataset) is solved. The following section provides an overview of how the code is organized. In particular, `pipelines/abalone/pipeline.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model.

A description of some of the artifacts is provided below:

This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CI/CD system (via GitHub Actions). This file has the fields defined for naming the Pipeline, ModelPackageGroup etc.

```
.github/workflows/build.yml
```


Pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. Since no instances for training were available for the free tier on AWS, the training is done through a "processing" step with an additional script. This is the core business logic.


```
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |   `-- train_processing.py

```

Utility modules for getting pipeline definition jsons and running pipelines (no need for further modification):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```

Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```

A stubbed testing module for testing the pipeline:
```
|-- tests
|   |-- test_endpoints.py
|   `-- test_pipelines.py
```

The `tox` testing framework configuration:
```
tox.ini
```

## Deploying the model

This code repository defines the CloudFormation template which defines the Endpoints as infrastructure. It has configuration files associated with `staging` and `prod` stages. 

Upon triggering a manual deployment, the GitHub Actions pipeline will deploy 2 Endpoints - `staging` and `prod`. After the first deployment is completed, GitHub waits for a manual approval step for promotion to the prod stage.

A description of some of the artifacts is provided below:

`.github/workflows/deploy.yml`  
  - Orchestrates the deployment steps (staging first; production after).

`build_deployment_configs.py`
  - Reads the base stage configs in this repo, `staging-config.json` and `prod-config.json`, and extends them at runtime with the latest ModelPackageArn, the execution role, and project tags.  
  It writes runtime files (`*-config-export.json`) that the deploy step consumes. 

`deploy_stack.py`
  - Takes the runtime export JSON (passed as `--param-file`) and creates/updates a CloudFormation stack using `endpoint-config-template.yml` that provisions the SageMaker Model, EndpointConfig, and Endpoint.

`endpoint-config-template.yml`
  - CloudFormation template used by the deploy workflow in each stage.

## Dataset for the Example Abalone Pipeline

The dataset used is the [UCI Machine Learning Abalone Dataset](https://archive.ics.uci.edu/ml/datasets/abalone) [1]. The aim for this task is to determine the age of an abalone (a kind of shellfish) from its physical measurements. At the core, it's a regression problem. 
 
The dataset contains several features - length (longest shell measurement), diameter (diameter perpendicular to length), height (height with meat in the shell), whole_weight (weight of whole abalone), shucked_weight (weight of meat), viscera_weight (gut weight after bleeding), shell_weight (weight after being dried), sex ('M', 'F', 'I' where 'I' is Infant), as well as rings (integer).

The number of rings turns out to be a good approximation for age (age is rings + 1.5). However, to obtain this number requires cutting the shell through the cone, staining the section, and counting the number of rings through a microscope -- a time-consuming task. However, the other physical measurements are easier to determine. The dataset is used to build a predictive model of the variable rings through these other physical measurements.

[1] Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.

## Outlook / TODO

- Auto-deploy on approval: When a new model version is Approved in the Model Package Group, automatically trigger the `DeploySageMakerModel` workflow (staging → prod).
  - Wire EventBridge → Lambda → GitHub Actions (`repository_dispatch`) or call `deploy_stack.py` directly from Lambda.
- Adding smoke/continuous test steps.
- Basic monitoring: Enable CloudWatch logs/metrics export for the endpoint.