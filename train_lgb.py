#import package
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from preprococessing import preprocessing_data
def train_model(path):

    # read data from csv


    print('Load Main Data')

    df = pd.read_csv(path, index_col = 0)
    df = preprocessing_data(df)

    # Split data into 80% of training and 20 %validation sets

    train_data = df[df['date'] < '2021-12-09']
    valid_data = df[df['date'] >= '2021-12-09']

    # Define LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 30,
        'learning_rate': 0.5,
        'n_estimators': 100
    }

    # Train LightGBM model
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        train_data.drop(
            ["day", "item_number", "item_name", "unit", "date",
             "sales_quantity"], axis = 1
            ), train_data['sales_quantity']
        )

    # Make predictions on validation set
    print('Training the ')
    valid_preds = model.predict(
        valid_data.drop(
            ["day", "item_number", "item_name", "unit", "date",
             'sales_quantity'], axis = 1
            )
        )

    # Round value
    valid_preds = np.round(valid_preds, decimals = 0)
    #

    # Evaluate model performance on validation set
    mse = mean_squared_error(valid_data['sales_quantity'], valid_preds)
    rmse = mse ** 0.5

    model_name = 'lgb_model.bin'
    pickle.dump(model, open(model_name, 'wb'))
    print(f'RMSE: {rmse}')
    print(f'model saved at: {model_name}')
