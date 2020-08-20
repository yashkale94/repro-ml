from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json
import random

df = pd.read_csv('data/Train_set.csv')
df = df[['0','1','2']]


samples = df.values

random.shuffle(samples)


X = samples[:,0:2]
y = samples[:,2]

X_train = X[0:100]
y_train = y[0:100]

X_test = X[100:]
y_test = y[100:]

LR = LogisticRegression()
LR.fit(X_train, y_train)

preds = LR.predict(X_test)

p = accuracy_score(y_test, preds)


with open('Metrics/metrics.json','w') as jsonfile:
    json.dump(dict(accuracy=p), jsonfile)
