import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

warnings.filterwarnings('ignore')

# import data
df= pd.read_csv('BitcoinHeistData.csv')

# remove low count labels
keep = ["white","paduaCryptoWall","montrealCryptoLocker","princetonCerber","princetonLocky","montrealCryptXXX"]
df = df[df.label.isin(keep) == True]

# downsample
df_maj = df[df.label == "white"]
df_p = df[df.label == "paduaCryptoWall"]
df_min = df[df.label != "white"]
df_min = df_min[df_min.label != "paduaCryptoWall"]
df_maj = df_maj.sample(n=12000, random_state=42)
df_p = df_p.sample(n=10000, random_state=42)

# recombine
df = pd.concat([df_maj, df_min, df_p])

# normalize
dfx = df
dfx = dfx.drop('label', axis=1)
dfx = dfx.drop('address', axis=1)
dfx = dfx.astype(float)
dfx.div(dfx.sum(axis=0), axis=1)
dfx.div(dfx.sum(axis=1), axis=0)
df = pd.concat([df['address'],dfx,df['label']],axis=1)

# print df info
df.info()
print(df.iloc[:, 9].value_counts(normalize=False))

# label to num
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=["str"]).columns:
    df[col] = label_encoder.fit_transform(df[col])

x = df.iloc[:,1:8].values
y = df.iloc[:,9].values

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

clf = svm.SVC(
    C=0.9,
    kernel='linear',
    max_iter=10000000,
    tol=0.00001,
    class_weight='balanced',
    verbose=2
)

clf.fit(X_train, y_train)

# results
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\n",classification_report(y_test, y_pred))
print("\n")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
