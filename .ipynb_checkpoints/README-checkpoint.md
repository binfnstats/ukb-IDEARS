# ukb-dementia-shap
 UKB dementia, AD and PD classification and SHAP

# Models to run

# IDEARS - Integrated Disease Explanation and Associations Risk Scoring

## Overview

This is the codebase for IDEARs - Integrated Disease Explanation and Associations Risk Scoring. Its overall architecture is shown below.

![](UKB ML flow-Page-2.drawio.png)

## How to Run
To ease the configuation, please install Anaconda and then use the conda-env.yml to create the required virutal environment. 

1. Install Anaconda:

https://www.anaconda.com/products/individual

2. Create the environment:

```conda env create -f .\conda-env.yml```

3. Acticate the environment:

```conda activate frollo-py-pipeline-env```

Then on Windows, run ```startlocal_woDocker.bat``` and on Linux, run ```startlocal_woDocker.sh```


## Codebase Structure

### Overview
Import modules etc.

### Directory Tree and Explanations

This folder has the implementation of this PoC. Due to the model and data access in PoC phase, the model used in the **hasmerch** endpoint uses a pre-defined hunggingface model as an example. It can be replaced to the model we are using easily once the model of interest is identified. 

```
📦frollo-py-inference-pipeline-poc
 ┣ 
 ┃ ┣ 📂core
 ┃ ┃ ┣ 📂logic
 ┃ ┃ ┃ ┣ 📜enrichment.py
 ┃ ┃ ┃ ┗ 📜hasmerch.py
 ┃ ┃ ┣ 📂models
 ┃ ┃ ┃ ┣ 📜bulkresponse.py
 ┃ ┃ ┃ ┣ 📜requests.py
 ┃ ┃ ┃ ┗ 📜response.py
 ┃ ┃ ┗ 📜config.py
 ┃ ┣ 📂model
 ┃ ┃ ┗ 📜README.md
 ┃ ┣ 
 ┣ 📜autoscale.yaml
 ┣ 📜conda-env.yml
 ┣ 📜Dockerfile
 ┣ 📜locustfile.py
 ┣ 📜main.py
 ┣ 📜README.md
 ┣ 📜PoC for Python Services Full.pdf
 ┣ 📜requirements.txt
 ┣ 📜startlocal_woDocker.bat
 ┗ 📜startlocal_woDocker.sh
```

### Root Folder
Some noticeable files are the following:
- conda-env.yml: conda environment definition for local use;

### Service Folder
By seperating api and core logic, this folder defines the actual endpoints as well as some core business logic. It also defines inputs and outputs. 

Some noticeable files are the following:

- core/logic/hasmerch.py: some essential business logic for preparing the data for further processing, such as sentence encoding and judgement about whether 
### Remarks on Data Required to Run

## Some Statistics

### ![](statistics.png)

## The Roadmap


## Get Help

Michael Allwright - michael@allwrightanalytics.com

