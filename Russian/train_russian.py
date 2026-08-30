import os
import shutil
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# RUSSIAN CYRILLIC HANDWRITING CNN
# ============================================================

print("=" * 70)
print("RUSSIAN CYRILLIC HANDWRITING CNN")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

ORIGINAL_DATASET = "img"
PROCESSED_DATASET = "img_processed"


# ============================================================
# CHECK ORIGINAL DATASET
# ============================================================

if not os.path.exists(ORIGINAL_DATASET):
    print("\nERROR: 'img' folder was not found.")
    print("Make sure this script is inside the Russian folder.")
    exit()

print("\nOriginal dataset found:")
print(os.path.abspath(ORIGINAL_DATASET))


# ============================================================
# CREATE PROCESSED DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: PREPARING IMAGES")
print("=" * 70)

print("\nOriginal dataset will NOT be modified.")
print("Creating processed dataset...")


# Delete previous processed copy if it exists
if os.path.exists(PROCESSED_DATASET):
    print("\nExisting img_processed folder found.")
    print("Removing old processed copy...")
    shutil.rmtree(PROCESSED_DATASET)

os.makedirs(PROCESSED_DATASET, exist_ok=True)


# ============================================================
# PROCESS IMAGES
# ============================================================

total_images = 0
processed_images = 0
failed_images = 0

class_folders = sorted(
    [
        folder
        for folder in os.listdir(ORIGINAL_DATASET)
        if os.path.isdir(os.path.join(ORIGINAL_DATASET, folder))
    ]
)

print("\nClasses found:", len(class_folders))


for class_name in class_folders:

    source_folder = os.path.join(
        ORIGINAL_DATASET,
        class_name
    )

    target_folder = os.path.join(
        PROCESSED_DATASET,
        class_name
    )

    os.makedirs(target_folder, exist_ok=True)

    class_count = 0

    print(f"\nProcessing class: {class_name}")

    for filename in os.listdir(source_folder):

        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        ):
            continue

        total_images += 1

        source_path = os.path.join(
            source_folder,
            filename
        )

        target_path = os.path.join(
            target_folder,
            filename
        )

        try:

            # --------------------------------------------
            # OPEN IMAGE
            # --------------------------------------------

            image = Image.open(source_path)

            # --------------------------------------------
            # HANDLE TRANSPARENCY
            # --------------------------------------------

            if image.mode == "RGBA":

                # White background
                background = Image.new(
                    "RGBA",
                    image.size,
                    (255, 255, 255, 255)
                )

                # Flatten transparent image onto white
                image = Image.alpha_composite(
                    background,
                    image
                )

                # Convert to grayscale
                image = image.convert("L")

            elif image.mode == "LA":

                background = Image.new(
                    "RGBA",
                    image.size,
                    (255, 255, 255, 255)
                )

                image = image.convert("RGBA")

                image = Image.alpha_composite(
                    background,
                    image
                )

                image = image.convert("L")

            else:

                # Convert any other format to grayscale
                image = image.convert("L")

            # --------------------------------------------
            # SAVE AS PNG
            # --------------------------------------------

            image.save(
                target_path,
                format="PNG"
            )

            processed_images += 1
            class_count += 1

        except Exception as e:

            failed_images += 1

            print(
                f"WARNING: Could not process {source_path}"
            )
            print(
                "Reason:",
                e
            )

    print(
        f"  Processed: {class_count} images"
    )


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    "\nTotal images found       :",
    total_images
)

print(
    "Successfully processed   :",
    processed_images
)

print(
    "Failed images            :",
    failed_images
)

if failed_images > 0:
    print(
        "\nWARNING: Some images could not be processed."
    )
else:
    print(
        "\n✅ All images processed successfully."
    )


# ============================================================
# DATA GENERATORS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: LOADING DATASET")
print("=" * 70)


dataset_path = PROCESSED_DATASET


# -----------------------------------
# TRAIN GENERATOR
# -----------------------------------

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

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

    rescale=1.0 / 255,

    validation_split=0.2
)


# ============================================================
# TRAIN DATA
# ============================================================

train_data = train_datagen.flow_from_directory(

    dataset_path,

    target_size=(64, 64),

    color_mode="grayscale",

    batch_size=32,

    class_mode="categorical",

    subset="training",

    shuffle=True
)


# ============================================================
# VALIDATION DATA
# ============================================================

val_data = val_datagen.flow_from_directory(

    dataset_path,

    target_size=(64, 64),

    color_mode="grayscale",

    batch_size=32,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

num_classes = len(
    train_data.class_indices
)

print("\nClasses found:")
print(
    train_data.class_indices
)

print(
    "\nTotal classes:",
    num_classes
)


# ============================================================
# SAVE LABELS
# ============================================================

index_to_label = {
    value: key
    for key, value in train_data.class_indices.items()
}


with open(
    "russian_labels.txt",
    "w",
    encoding="utf-8"
) as f:

    for i in range(num_classes):

        f.write(
            index_to_label[i] + "\n"
        )


print(
    "\n✅ Labels saved as russian_labels.txt"
)


# ============================================================
# CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: BUILDING CNN")
print("=" * 70)


model = models.Sequential([

    # -----------------------------------
    # CONVOLUTION BLOCK 1
    # -----------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(64, 64, 1)
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # -----------------------------------
    # CONVOLUTION BLOCK 2
    # -----------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # -----------------------------------
    # CONVOLUTION BLOCK 3
    # -----------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # -----------------------------------
    # CONVOLUTION BLOCK 4
    # -----------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),

    layers.BatchNormalization(),


    # -----------------------------------
    # CLASSIFIER
    # -----------------------------------

    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: TRAINING")
print("=" * 70)

print(
    "\nTraining for up to 25 epochs..."
)

print(
    "Early stopping is enabled."
)


history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=25,

    callbacks=[
        early_stop,
        reduce_lr
    ]
)


# ============================================================
# SAVE ONLY H5
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: SAVING MODEL")
print("=" * 70)


model.save(
    "russian_model.h5"
)


print(
    "\n✅ Russian model saved as:"
)

print(
    "russian_model.h5"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("RUSSIAN MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nCreated:")

print("  ✅ russian_model.h5")
print("  ✅ russian_labels.txt")
print("  ✅ img_processed/")

print(
    "\nOriginal img/ dataset was NOT modified."
)

print("=" * 70)
