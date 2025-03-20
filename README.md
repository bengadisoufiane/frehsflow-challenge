# Freshflow challenge


## install requirement

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirement file.

```bash
pip install requirements.txt
```
## Download the data

Download the [data](https://hackmd.io/@freshflow/B16yJrWg9#Input) from this link  in but it in the data folder

## Analysis & feature engineering

The Jupyter Notebook provides an overview of the data and presents various feature engineering techniques to enhance the data quality and enrich the model's input.

## To train the models
### LGB MODEL

```bash
python3 train_lgb.py data/data.csv 
```

### XGBOOST MODEL

```bash
python3 train_xgboost.py data/data.csv 
```

## To run  the models
### LightGBM MODEL

```bash
python3 prediction_lgb.py data/data.csv  model/lgb_model.bin
```

### XGBOOST MODEL

```bash
python3 prediction_xgboost.py data/data.csv  model/xgbost_model.bin
```


## The output

The model output is the RMSE of the model and the path of saved model


## Results

* RMSE for LightGBM model is : 4.266
* RMSE for XGBOOST model is  : 4.346

The LGB model is slightly better than the XGBOOST MODEL
##  For more information
 For any questions, please reach out to bengadisoufiane@gmail.com.
