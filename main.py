import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classify import Classifiers


data_path = 'data/03-11/UDPLag.csv'
data = pd.read_csv(data_path)

redundant_col = ['Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port',
                 ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp', 'SimillarHTTP']
data.drop(axis=1, columns=redundant_col, inplace=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna(how="any")

X_dos = data.loc[data[' Label'] == 'Syn',
                 :' Inbound'].values[:5000].astype(np.float)
y_dos = np.ones(X_dos.shape[0])
X_beg = data.loc[data[' Label'] == 'BENIGN',
                 :' Inbound'].values.astype(np.float)
y_beg = np.zeros(X_beg.shape[0])

X = np.concatenate((X_dos, X_beg))
y = np.concatenate((y_dos, y_beg))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
x_max = -np.inf
for row in X_train:
    x_max = max(x_max, np.max(row))
print(x_max)

classifiers = Classifiers()
report, roc = classifiers.run(X_train, X_test, y_train, y_test)

print(report)
with open('result/report.txt', 'w') as f:
    f.write(report)
roc.savefig('result/ROC.png')
