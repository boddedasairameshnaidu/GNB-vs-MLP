import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pickle

def trainModel():
    file = "hypothyroid.csv"
    df = pd.read_csv(file)
    df = df.dropna()
    df = df.drop_duplicates()
    alg = MLPClassifier()

    y = df['result']
    x = df.drop(['result'],1)

    model = alg.fit(x,y)

    pickle.dump(model, open('modelMLP.sav','wb'))

trainModel()

    