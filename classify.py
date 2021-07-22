import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

SEED = 42

CLASSIFIERS = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=SEED),
    'Random Forest': RandomForestClassifier(random_state=SEED, n_jobs=-1),
    'SVM': SVC(random_state=SEED, probability=True),
}
HYPER_GRID = {
    'K-Nearest Neighbors': {"n_neighbors": [10, 100, 1000]},
    'Decision Tree': {"criterion": ["gini", "entropy"]},
    'Random Forest': {"n_estimators": [10, 100, 1000]},
    'SVM': {"C": np.logspace(-1, 1, 3), "gamma": np.logspace(-1, 1, 3)},

}
COLORS = ['purple', 'orange', 'green', 'red']


class Classifiers:
    def __init__(self):
        self.classifiers = {}
        for name in CLASSIFIERS:
            self.classifiers[name] = GridSearchCV(
                CLASSIFIERS[name],
                HYPER_GRID[name],
                cv=5,
                n_jobs=-1
            )

    def run(self, X_train, X_test, y_train, y_test):
        report = f'Original data size: {X_train.shape} {X_test.shape}\n'

        # Perform feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Perform feature selection
        ftsl = SelectFromModel(
            LinearSVC(penalty="l1", dual=False, random_state=SEED).fit(X_train, y_train), prefit=True)
        X_train = ftsl.transform(X_train)
        X_test = ftsl.transform(X_test)
        report += f'Processed data size: {X_train.shape} {X_test.shape}\n'

        roc_auc = {}
        for name in self.classifiers:
            print(f'Running {name}... ', end='', flush=True)
            self.classifiers[name].fit(X_train, y_train)

            y_prob = self.classifiers[name].predict_proba(X_test)[:, 1]
            y_pred = self.classifiers[name].predict(X_test)
            print('%.4f' % metrics.accuracy_score(y_test, y_pred))

            # Generate classification report
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cnf_matrix.ravel()
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
            auc = metrics.roc_auc_score(y_test, y_prob)
            other_metrics = pd.DataFrame({
                'TPR': '%.4f' % TPR,
                'FPR': '%.4f' % FPR,
                'ROC AUC': '%.4f' % auc,
            }, index=[0]).to_string(col_space=9, index=False)
            roc_auc[name] = [auc, fpr, tpr]
            report += '-' * 80 + '\n'
            report += name + '\n'
            report += f'{metrics.classification_report(y_test, y_pred, digits=4)}\n'
            report += f'{cnf_matrix}\n\n'
            report += f'{other_metrics}\n'

        # Draw ROC curve
        roc = self.draw_roc(roc_auc)
        return report, roc

    def draw_roc(self, roc_auc):
        lw = 2
        roc_auc = dict(sorted(roc_auc.items(), key=lambda k: k[1][0]))
        plt.figure()
        for name, color in zip(roc_auc, COLORS):
            auc, fpr, tpr = roc_auc[name]
            plt.plot(fpr, tpr, color=color, lw=lw,
                     label="%s (AUC = %0.4f)" % (name, auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("1-Specificity(False Positive Rate)")
            plt.ylabel("Sensitivity(True Positive Rate)")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
        return plt
