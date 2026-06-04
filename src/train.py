from preprocess import get_data_generators
from model import build_model
import pickle

train_gen, val_gen = get_data_generators("data/raw")

model = build_model(train_gen.num_classes)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

model.save("models/model.keras")

# Save class names
with open("models/classes.pkl", "wb") as f:
    pickle.dump(train_gen.class_indices, f)