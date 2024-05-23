import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import pickle

def trainModel():
    file = "hypothyroid.csv"
    df = pd.read_csv(file)
    df = df.dropna()
    df = df.drop_duplicates()
    alg = GaussianNB()

    y = df['result']
    x = df.drop(['result'],1)

    model = alg.fit(x,y)

    pickle.dump(model, open('model.sav','wb'))

trainModel()

    