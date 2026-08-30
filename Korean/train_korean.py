import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------------
# DATASET PATH
# -----------------------------------

dataset_path = "img"

# -----------------------------------
# TRAIN GENERATOR
# -----------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

# -----------------------------------
# VALIDATION GENERATOR
# -----------------------------------

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# -----------------------------------
# TRAIN DATA
# -----------------------------------

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# -----------------------------------
# VALIDATION DATA
# -----------------------------------

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -----------------------------------
# CLASS INFO
# -----------------------------------

num_classes = len(train_data.class_indices)

print("Classes found:")
print(train_data.class_indices)

print("Total classes:", num_classes)

# -----------------------------------
# MODEL
# -----------------------------------

model = models.Sequential([

    layers.Conv2D(
        32, (3,3),
        activation='relu',
        input_shape=(64,64,1)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation='softmax')
])

# -----------------------------------
# COMPILE
# -----------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------
# CALLBACKS
# -----------------------------------

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

# -----------------------------------
# TRAIN
# -----------------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop, reduce_lr]
)

# -----------------------------------
# SAVE
# -----------------------------------

model.save("korean_model.h5")

print("Korean model training complete!")
print("Model saved as korean_model.h5")
print("Labels saved as labels.txt")
