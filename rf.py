import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

df= pd.read_csv('BitcoinHeistData.csv')

print(df.iloc[:, 9].value_counts(normalize=False))

# remove small classes
keep = ["white","paduaCryptoWall","montrealCryptoLocker",
        "princetonCerber","princetonLocky","montrealCryptXXX",
        "montrealNoobCrypt","montrealDMALockerv3","montrealDMALocker"]
df = df[df.label.isin(keep) == True]

# downsample "white"
df_maj = df[df.label == "white"]
df_min = df[df.label != "white"]
df_maj = df_maj.sample(n=len(df_min), random_state=42)

# upsample fewest
min_min = ["montrealNoobCrypt","montrealDMALockerv3","montrealDMALocker"]
df_min_min = df[df.label.isin(min_min) == True]
df_min_min = df_min_min.sample(n=len(df_min), replace = True, random_state=42)

# recombine
df = pd.concat([df_maj, df_min, df_min_min])

df.info()

print(df.iloc[:, 9].value_counts(normalize=False))

label_encoder = LabelEncoder()

for col in df.select_dtypes(include=["str"]).columns:
    df[col] = label_encoder.fit_transform(df[col])

x = df.iloc[:,1:9].values
y = df.iloc[:,9].values

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=0,
    n_jobs=-1,
    verbose=2,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# results
y_pred = rf.predict(X_test)
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
