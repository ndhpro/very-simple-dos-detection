import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from classify import Classifiers


data_path = 'data/final dataset.arff'
data = arff.loadarff(data_path)
data = pd.DataFrame(data[0])

redundant_col = ['SRC_ADD', 'DES_ADD', 'PKT_TYPE',
                 'FLAGS', 'NODE_NAME_FROM', 'NODE_NAME_TO']
data.drop(axis=1, columns=redundant_col, inplace=True)
data = data.replace([np.inf, -np.inf], np.nan).dropna(how="any")

X_dos = data.loc[data['PKT_CLASS'] == b'UDP-Flood',
                 :'LAST_PKT_RESEVED'].values[:10000].astype(np.float)
y_dos = np.ones(X_dos.shape[0])
X_beg = data.loc[data['PKT_CLASS'] == b'Normal',
                 :'LAST_PKT_RESEVED'].values[:10000].astype(np.float)
y_beg = np.zeros(X_beg.shape[0])
print(X_dos.shape, X_beg.shape)

X = np.concatenate((X_dos, X_beg))
y = np.concatenate((y_dos, y_beg))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

classifiers = Classifiers()
report, roc = classifiers.run(X_train, X_test, y_train, y_test)

print(report)
with open('result/report.txt', 'w') as f:
    f.write(report)
roc.savefig('result/ROC.png')
