import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

model = pickle.load(open('modelMLP.sav','rb'))

file = "hypothyroid_test.csv"
df = pd.read_csv(file)
y = df['result']
del df['result']
predicted_data = model.predict(df)

acc = accuracy_score(y,predicted_data)
pre = precision_score(y,predicted_data,average='micro')
rs = recall_score(y,predicted_data,average='micro')
fs = f1_score(y,predicted_data,average='micro')

print("Accuracy: ",acc)

labels = ['Accuracy','Precision',"Recall","F1_score"]
val = [acc,pre,rs,fs]
plt.bar(labels,val,color=['red','green','blue','yellow'])
plt.xlabel("PERFORMANCE METRIC")
plt.ylabel("SCORE")
plt.title("MLP CLASSIFIER PERFORMANCE")
plt.show()
