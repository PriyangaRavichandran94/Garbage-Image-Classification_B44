import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model("models/best_model.h5")

with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=(224,224),
    batch_size=32
)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))