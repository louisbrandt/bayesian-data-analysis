# imports 
import pandas as pd 
import numpy as np

def process_revenue_data(revenue_df,gbs=None):

    revenue_df['date'] = pd.to_datetime(revenue_df['date'])

    # define index variables
    revenue_df['dow'] = revenue_df['date'].dt.weekday # day of week as integer 0-6
    revenue_df['day'] = revenue_df['date'].dt.day - 1
    revenue_df['month'] = revenue_df['date'].dt.month - 1
    revenue_df['year'] = revenue_df['date'].dt.year - 2020

    # if test is True, return the normalized revenue based on the mean and std of the training data
    if gbs is not None:
        Y_MEAN, Y_STD = gbs
        revenue_df['revenue'] = revenue_df['revenue'].apply(lambda x: (x - Y_MEAN) / Y_STD)
        revenue_df['revenue_diff'] = revenue_df['revenue'].diff()
        return revenue_df[['date','n_stores','dow','day','month','year','revenue','revenue_diff']]
    
    # when revenue is less than 61000, replace the revenue with the mean of the previous 7 days and the next 7 days
    for i in range(len(revenue_df)):
        if revenue_df.loc[i,'revenue'] < 61000:
            revenue_df.loc[i,'revenue'] = np.mean(revenue_df.loc[i-7:i+7,'revenue'])

    # calculate mean and std of revenue
    Y_STD = revenue_df['revenue'].std()
    Y_MEAN = revenue_df['revenue'].mean()

    revenue_df['revenue'] = revenue_df['revenue'].apply(lambda x: (x - Y_MEAN) / Y_STD)
    revenue_df['revenue_diff'] = revenue_df['revenue'].diff()
    return revenue_df[['date','n_stores','dow','day','month','year','revenue','revenue_diff']], (Y_MEAN, Y_STD)

def process_weather_data(weather_df,gbs=None):
    # normalise weather features
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    # turn precip into categorical variable with 3 categories 0, <1 and >=1
    weather_df['precip'] = weather_df['precip'].apply(lambda x: 0 if x == 0 else 1 if x < 1 else 2)

    if gbs is not None:
        HUMIDITY_MEAN, HUMIDITY_STD, TEMP_MAX_MEAN, TEMP_MAX_STD, CLOUD_MEAN, CLOUD_STD, WIND_MEAN, WIND_STD = gbs
        # return the normalized weather features based on the mean and std of the training data
        weather_df['humidity'] = weather_df['humidity'].apply(lambda x: (x - HUMIDITY_MEAN) / HUMIDITY_STD)
        weather_df['tempmax'] = weather_df['tempmax'].apply(lambda x: (x - TEMP_MAX_MEAN) / TEMP_MAX_STD)
        weather_df['cloudcover'] = weather_df['cloudcover'].apply(lambda x: (x - CLOUD_MEAN) / CLOUD_STD)
        weather_df['windspeed'] = weather_df['windspeed'].apply(lambda x: (x - WIND_MEAN) / WIND_STD)
        return weather_df[['date','humidity','tempmax','precip','cloudcover','windspeed']]

    # otherwise define the mean and std of the weather features
    HUMIDITY_MEAN = weather_df['humidity'].mean()
    HUMIDITY_STD = weather_df['humidity'].std()
    TEMP_MAX_MEAN = weather_df['tempmax'].mean()
    TEMP_MAX_STD = weather_df['tempmax'].std()
    CLOUD_MEAN = weather_df['cloudcover'].mean()
    CLOUD_STD = weather_df['cloudcover'].std()
    WIND_MEAN = weather_df['windspeed'].mean()
    WIND_STD = weather_df['windspeed'].std()
    # and then normalize the weather features
    weather_df['humidity'] = weather_df['humidity'].apply(lambda x: (x - HUMIDITY_MEAN) / HUMIDITY_STD)
    weather_df['tempmax'] = weather_df['tempmax'].apply(lambda x: (x - TEMP_MAX_MEAN) / TEMP_MAX_STD)
    weather_df['cloudcover'] = weather_df['cloudcover'].apply(lambda x: (x - CLOUD_MEAN) / CLOUD_STD)
    weather_df['windspeed'] = weather_df['windspeed'].apply(lambda x: (x - WIND_MEAN) / WIND_STD)
    return weather_df[['date','humidity','tempmax','cloudcover','windspeed','precip']], (HUMIDITY_MEAN, HUMIDITY_STD, TEMP_MAX_MEAN, TEMP_MAX_STD, CLOUD_MEAN, CLOUD_STD, WIND_MEAN, WIND_STD)

def join_revenue_weather(revenue_df, weather_df):
    df = pd.merge(revenue_df,weather_df,on='date')
    # convert all index variables to categorical
    df = df.astype({'precip':'category','n_stores':'category','dow':'category','day':'category','month':'category','year':'category'})
    return df[['date','n_stores','dow','day','month','year','humidity','tempmax','cloudcover','windspeed','precip','revenue','revenue_diff']]
