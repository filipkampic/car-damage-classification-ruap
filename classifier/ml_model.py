import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
from ml.scripts.predict_single_image import extract_hog_features_from_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LDA_MODEL_PATH = os.path.join(BASE_DIR, "ml", "data", "lda_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "ml", "data", "rf_model.pkl")

lda = joblib.load(LDA_MODEL_PATH)
rf = joblib.load(RF_MODEL_PATH)

def predict_from_image_path(image_path):
    hog_vec = extract_hog_features_from_image(image_path)
    hog_vec = hog_vec.reshape(1, -1)
    lda_vec = lda.transform(hog_vec)
    pred = rf.predict(lda_vec)[0]
    return pred
