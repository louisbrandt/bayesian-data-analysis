import argparse
from functions import *

def process_train(rev, wea):
    # --- prepare data

    print('Processing data...')
    rev, r_gbs = process_revenue_data(rev)
    wea, w_gbs = process_weather_data(wea)
    
    print('Joining data...')
    df = join_revenue_weather(rev,wea)

    # --- write df to csv
    df.to_csv('/Users/louisbrandt/itu/6/bachelor/data/processed_data.csv',index=False)
    print('Data written to csv.')

    # --- write globas to csv 
    with open('/Users/louisbrandt/itu/6/bachelor/data/globals.csv','w') as f:
        f.write(f'{r_gbs[0]},{r_gbs[1]},{w_gbs[0]},{w_gbs[1]},{w_gbs[2]},{w_gbs[3]},{w_gbs[4]},{w_gbs[5]},{w_gbs[6]},{w_gbs[7]}')
    print('Globals written to csv.')

def process_test(rev, wea):

    # --- load globals
    with open('/Users/louisbrandt/itu/6/bachelor/data/globals.csv','r') as f:
        data = f.read()
        data = data.split(',')
        Y_MEAN = float(data[0])
        Y_STD = float(data[1])
        HUMIDITY_MEAN = float(data[2])
        HUMIDITY_STD = float(data[3])
        TEMP_MAX_MEAN = float(data[4])
        TEMP_MAX_STD = float(data[5])
        CLOUD_MEAN = float(data[6])
        CLOUD_STD = float(data[7])
        WIND_MEAN = float(data[8])
        WIND_STD = float(data[9])
    print('Globals loaded from csv.')

    print('Processing data...')
    rev = process_revenue_data(rev,gbs=(Y_MEAN, Y_STD))
    wea = process_weather_data(wea,gbs=(HUMIDITY_MEAN, HUMIDITY_STD, TEMP_MAX_MEAN, TEMP_MAX_STD, CLOUD_MEAN, CLOUD_STD, WIND_MEAN, WIND_STD))

    print('Joining data...')
    df = join_revenue_weather(rev,wea)

    # --- write df to csv
    print('Data written to csv.')
    df.to_csv('data/processed_test_data.csv',index=False)

def unnormalise_data(df):
    # load globals
    with open('/Users/louisbrandt/itu/6/bachelor/data/globals.csv','r') as f:
        data = f.read()
        data = data.split(',')
        Y_MEAN = float(data[0])
        Y_STD = float(data[1])
        HUMIDITY_MEAN = float(data[2])
        HUMIDITY_STD = float(data[3])
        TEMP_MAX_MEAN = float(data[4])
        TEMP_MAX_STD = float(data[5])
        CLOUD_MEAN = float(data[6])
        CLOUD_STD = float(data[7])
        WIND_MEAN = float(data[8])
        WIND_STD = float(data[9])
    
    # unnormalise data
    df['revenue'] = df['revenue'] * Y_STD + Y_MEAN
    df['humidity'] = df['humidity'] * HUMIDITY_STD + HUMIDITY_MEAN
    df['tempmax'] = df['tempmax'] * TEMP_MAX_STD + TEMP_MAX_MEAN
    df['cloudcover'] = df['cloudcover'] * CLOUD_STD + CLOUD_MEAN
    df['windspeed'] = df['windspeed'] * WIND_STD + WIND_MEAN
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    # parse train or test argument
    parser.add_argument('mode', type=str, help='Mode to use', choices=['train', 'test'])
    # parse revenue data path argument 
    parser.add_argument('--rev', type=str, help='Path to revenue data', default='/Users/louisbrandt/itu/6/bachelor/data/revenue_data.csv')
    # parse weather data path argument 
    parser.add_argument('--wea', type=str, help='Path to weather data', default='/Users/louisbrandt/itu/6/bachelor/data/weather_data.csv')
    args = parser.parse_args()
    
    rev = pd.read_csv(args.rev)
    wea = pd.read_csv(args.wea)

    if args.mode == 'train':
        process_train(rev, wea)
    elif args.mode == 'test':
        process_test(rev, wea)

