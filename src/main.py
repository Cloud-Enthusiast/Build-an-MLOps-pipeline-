from data_preprocessing import data_preprocessing
from feature_engineering import feature_engineering
from model_selection import model_selection

df = data_preprocessing('D:\90 days DSA Goat\ML journey\Build-an-MLOps-pipeline-\Indians screentime\Indian_Kids_Screen_Time.csv', "Health_Impacts")

fe = feature_engineering(df,['Gender', 'Primary_Device'] ,['Urban_or_Rural'], ['Gender_Female', 'Primary_Device_Tablet'])

print(model_selection(fe[0], fe[1]))