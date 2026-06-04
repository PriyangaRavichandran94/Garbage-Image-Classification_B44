from tensorflow.keras.models import load_model
from preprocess import get_data_generators
from sklearn.metrics import classification_report
import numpy as np

model = load_model("models/model.h5")

_, val_gen = get_data_generators("data/raw")

preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)

print(classification_report(val_gen.classes, y_pred))