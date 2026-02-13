import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

HOG_CSV_PATH = r"C:\Users\Korisnik\Projekti\CarDamageSeverityRUAP\carapp\ml\data\hog_optimized.csv"
LDA_MODEL_PATH = r"C:\Users\Korisnik\Projekti\CarDamageSeverityRUAP\carapp\ml\data\lda_model.pkl"
RF_MODEL_PATH = r"C:\Users\Korisnik\Projekti\CarDamageSeverityRUAP\carapp\ml\data\rf_model.pkl"

print("Učitavam HOG CSV...")
df = pd.read_csv(HOG_CSV_PATH)
print("CSV shape:", df.shape)

X = df.drop(columns=['label']).values
y = df['label'].values

print("Učitavam LDA model...")
lda = joblib.load(LDA_MODEL_PATH)

print("Radim LDA transformaciju...")
X_lda = lda.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_lda, y, test_size=0.2, random_state=42
)

print("Treniram Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)

print("Evaluacija...")
pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

print("Spremam RF model...")
joblib.dump(rf, RF_MODEL_PATH)
print("Gotovo! RF model spremljen u:", RF_MODEL_PATH)
