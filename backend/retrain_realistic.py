import sys, os
sys.path.append(os.path.abspath('.'))
from app.training.data_generator import build_realistic_dataset   # NEW
from app.models.audio_models import create_cough_classifier, get_callbacks
import tensorflow as tf



# ---- load realistic data ----
X_train, X_val, y_train, y_val = build_realistic_dataset(n_per_class=1200)

# ---- build model with dropout + label smoothing ----
model = create_cough_classifier(input_shape=(128, 129, 1), num_classes=3)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ---- train ----
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=get_callbacks('cough_classifier_realistic'),
    verbose=2
)

print("New realistic model saved → models/cough_classifier_realistic.h5")