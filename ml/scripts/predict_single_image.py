import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel
from skimage import io, color, transform

HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (12, 12),
    'cells_per_block': (2, 2),
    'visualize': False,
    'feature_vector': True,
    'block_norm': 'L2-Hys'
}

TARGET_SIZE = (128, 128)

LDA_MODEL_PATH = r"C:\Users\Korisnik\Projekti\CarDamageSeverityRUAP\carapp\ml\data\lda_model.pkl"
RF_MODEL_PATH = r"C:\Users\Korisnik\Projekti\CarDamageSeverityRUAP\carapp\ml\data\rf_model.pkl"

lda = joblib.load(LDA_MODEL_PATH)
rf = joblib.load(RF_MODEL_PATH)

def extract_hog_features_from_image(image_path):
    img = io.imread(image_path)
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img

    img_resized = transform.resize(img_gray, TARGET_SIZE)

    hog_features = hog(img_resized, **HOG_PARAMS)

    edges = sobel(img_resized)
    edge_density = np.mean(edges > 0.1)

    brightness = np.mean(img_resized)
    contrast = np.std(img_resized)

    lbp = local_binary_pattern(img_resized, P=8, R=1, method='uniform')
    texture = np.std(lbp)

    if img.ndim == 3:
        color_diff = np.mean(np.abs(img[:,:,0] - img[:,:,1])) + \
                     np.mean(np.abs(img[:,:,1] - img[:,:,2]))
    else:
        color_diff = 0.0

    feature_vector = np.concatenate([
        hog_features,
        [edge_density, brightness, contrast, texture, color_diff]
    ])

    return feature_vector

def predict_image(image_path):
    hog_vec = extract_hog_features_from_image(image_path)
    hog_vec = hog_vec.reshape(1, -1)

    lda_vec = lda.transform(hog_vec)
    pred = rf.predict(lda_vec)[0]

    return pred

if __name__ == "__main__":
    test_image = r"C:\Users\Korisnik\Projekti\CarDamageSeverityRUAP\carapp\ml\data\0018.JPEG"
    label = predict_image(test_image)
    print("Predikcija:", label)
