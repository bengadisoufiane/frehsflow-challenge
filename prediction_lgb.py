import sys
import pandas as pd
import numpy as np
import pickle

import lightgbm as lgb
from preprocessing import preprocessing_data
def predict_using_model(file_path,model_path):

    # read data from csv


    print('Load Main Data')

    df = pd.read_csv(file_path, index_col = 0)
    df = preprocessing_data(df)

    # Load trained XGBoost model
    # Load trained XGBoost model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)


    # Make predictions on validation set
    print('Training the ')
    preds = model.predict(
        df.drop(
            ["day", "item_number", "item_name", "unit", "date",
             'sales_quantity'], axis = 1
            )
        )

    # Round value
    preds = np.round(preds, decimals = 0)
    print(preds)


if __name__ == '__main__':
    # Get the file path from the command line arguments
    if len(sys.argv) != 3:
        print("Usage: python script_name.py file_path.csv or model_path")
    else:
        file_path = sys.argv[1]
        model_path = sys.argv[2]
        predict_using_model(file_path,model_path)