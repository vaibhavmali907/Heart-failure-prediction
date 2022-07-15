import pickle
import pandas as pd

model = pickle.load(open('classifier.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))
class_names = [0,1]

def predict(df):
    df = df[['ejection_fraction','serum_creatinine','serum_sodium','time']]
    df = sc.transform(df)
    predictions = model.predict(df)
    output = [class_names[class_predicted] for class_predicted in predictions]
    return output
