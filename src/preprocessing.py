import pandas as pd

def load_energy_data(path:str):

    df = pd.read_csv(path)
    df['Datetime']=pd.to_datetime(df["Datetime"])
    df.set_index("Datetime",inplace=True)
    df.sort_index(inplace=True)

    return df


def clean_and_resample(df:pd.DataFrame, freq:str = "D"):
    df = df.resample(freq).mean()
    df.interpolate(method="time",nplace=True)

    return df

def train_test_split(df, test_size:int=30):
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train,test