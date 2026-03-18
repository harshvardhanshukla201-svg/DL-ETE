import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# -----------------------------
# DATA PREPROCESSING (CORRECTED)
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)
print("Class indices:", train_data.class_indices)
val_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# CLASS WEIGHTS (IMPORTANT FOR SMALL DATA)
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

# -----------------------------
# MODEL (TRANSFER LEARNING)
# -----------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Fine-tune last layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# -----------------------------
# CUSTOM HEAD
# -----------------------------
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(train_data.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop],
    class_weight=class_weights   # 🔥 important
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("food_model_advanced.keras")

print("Advanced model saved successfully!")