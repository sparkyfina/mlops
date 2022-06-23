
import pickle
import pandas as pd
import argparse



def read_data(filename):
    print("reading file name = ", filename)
    df = pd.read_parquet(filename)

    categorical = ['PUlocationID', 'DOlocationID']
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def run(year, month):

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']
    
    #df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet')
    #df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-03.parquet')
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month:02d}.parquet')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("The predicted duration mean = ", y_pred.mean())


    # write the ride id and the predictions to a dataframe with results.
    #process_date = datetime.strptime(date_in, "%Y-%m-%d")
    #month = process_date.month
    #year = process_date.year

    #month = 2
    #year = 2021

    print(f"year = {year} type {type(year)} and month = {month} with type {type(month)}")
   
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    #df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')

    #df_result = df['ride_id']   
    #df_result['Duration'] = y_pred.tolist()
    df_result = df[['ride_id', 'duration']]

    print(df_result.head())

    output_file = "df.parquet"
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="year of FVH data"
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="year of FVH data"
    )
    
    args = parser.parse_args()

    run(args.year, args.month)

