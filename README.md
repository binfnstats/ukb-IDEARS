# ukb-dementia-shap
 UKB dementia, AD and PD classification and SHAP

# Models to run

# IDEARS - Integrated Disease Explanation and Associations Risk Scoring

## Overview

This is the codebase for IDEARs - Integrated Disease Explanation and Associations Risk Scoring. Its overall architecture is shown below.


<img src="UKB ML flow-Page-2.drawio.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />


## How to Run
To ease the configuation, please install Anaconda and set this up in a virtual environment. 

1. Install Anaconda:

https://www.anaconda.com/products/individual

2. Create the environment:

```conda env create -f .\conda-env.yml```

3. Acticate the environment:

```conda activate conda-env```

Then on Windows, run ```startlocal_woDocker.bat``` and on Linux, run ```startlocal_woDocker.sh```


## Codebase Structure

data_gen.py is used to perform ETL on the data and to create the model datasets
data_proc.py is used for extra data processing including the creation of normalised datasets
ml.py is used to run the models including logistic regression, XGBoost and for model interpretability using SHAP
analysis.py is used to create charts, perform extra statistical tests including paired t tests

### Overview
Import modules etc.

### Directory Tree and Explanations

This folder shows the implementation of the IDEARs platform.


## Enquiries

Michael Allwright - michael@allwrightanalytics.com

