import pandas as pd
import logger

# Loading the dataset to create a DataFrame
try:
    def data_preprocessing(file_path:str, irrelevant_columns):
        df = pd.read_csv(file_path)

        # print(df.head(), "\n")
        df = df.drop(irrelevant_columns, axis='columns')
        # print("\n",df.isna().sum())

        return df

except FileNotFoundError:
    logger.error('File not found')
    raise

# print(data_preprocessing('D:\90 days DSA Goat\ML journey\Build-an-MLOps-pipeline-\Indians screentime\Indian_Kids_Screen_Time.csv', 'Health_Impacts'))