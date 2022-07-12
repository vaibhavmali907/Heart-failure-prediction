import pickle
import pandas as pd

model = pickle.load(open('classifier.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))
class_names = [0,1]

def predict(df):
    df = df[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
    df = sc.transform(df)
    predictions = model.predict(df)
    output = [class_names[class_predicted] for class_predicted in predictions]
    return output