import pandas as pd
import numpy as np



def preprocessing_data(df):
    # drop duplicated data
    df = df.drop_duplicates()
    # sort values
    df = df.sort_values('day').reset_index()

    # Handle null values
    df['revenue'] = df['revenue'].fillna(0)

    # create a regular expression to match the weight and unit of each product
    regex = r'(\d+)(G|KG)'

    # extract the weight and unit using the regular expression
    df['weight'] = df['item_name'].str.extract(regex, expand = False)[0].astype(
        int
        )
    df['unit'] = df['item_name'].str.extract(regex, expand = False)[1]
    # Perform one-hot encoding
    one_hot = pd.get_dummies(df['unit'], prefix = 'unit')
    df = pd.concat([df, one_hot], axis = 1)

    # sales price
    df['sales_price'] = df['revenue'] / df['sales_quantity']
    # handle nan value in sales price
    df['sales_price'] = df['sales_price'].fillna(df['suggested_retail_price'])
    # Orders quantity at the beginnin of the day
    df['orders_quantity'] = df[['orders_quantity', 'sales_quantity']].max(
        axis = 1
        )
    # quantity at the end of the day
    df['end_quantity'] = df['orders_quantity'] - df['sales_quantity']
    # discount
    df['discount'] = (df['sales_price'] - df['suggested_retail_price']) / df[
        'suggested_retail_price'] * 100

    # We can do some basic aggregations
    df['price_max'] = df.groupby(['item_number'])['sales_price'].transform(
        'max'
        )
    df['price_min'] = df.groupby(['item_number'])['sales_price'].transform(
        'min'
        )
    df['price_std'] = df.groupby(['item_number'])['sales_price'].transform(
        'std'
        )
    df['price_mean'] = df.groupby(['item_number'])['sales_price'].transform(
        'mean'
        )

    # Convert to DateTime
    df['date'] = pd.to_datetime(df['day'])

    # Make some features from date
    df['tm_d'] = df['date'].dt.day.astype(np.int8)
    df['tm_w'] = df['date'].dt.week.astype(np.int8)
    df['tm_m'] = df['date'].dt.month.astype(np.int8)
    df['tm_y'] = df['date'].dt.year
    df['tm_y'] = (df['tm_y'] - df['tm_y'].min()).astype(np.int8)

    df['tm_dw'] = df['date'].dt.dayofweek.astype(np.int8)
    df['tm_w_end'] = (df['tm_dw'] >= 5).astype(np.int8)
    return df