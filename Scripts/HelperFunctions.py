import os
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, regularizers

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow is using GPU: {gpus}")
    else:
        print("TensorFlow is not using a GPU.")

def load_and_decode_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    return image

def random_blur_image(image):
    sigma = np.random.uniform(50.0, 100.0)  
    filter_shape = (np.random.randint(10, 30), np.random.randint(10, 30))  
    image = tf.expand_dims(image, axis=0)
    blurred_image = tfa.image.gaussian_filter2d(image, filter_shape=filter_shape, sigma=sigma)
    blurred_image = tf.squeeze(blurred_image, axis=0)
    return blurred_image

def display_img(axes, image, i, side, title):
    axes[i, side].imshow(image)
    axes[i, side].axis('off')
    axes[i, side].set_title(title)

def show_images_side_by_side(clear_images, blurred_images, num_images=5):
    fig, axes = plt.subplots(num_images, 2, figsize=(5, 2 * num_images))
    for i in range(num_images):
        clear_image = clear_images[i].numpy()
        blurred_image = blurred_images[i].numpy()
        display_img(axes, clear_image, i, 0, 'Clear Image')
        display_img(axes, blurred_image, i, 1, 'Blurred Image')
    plt.tight_layout()
    plt.show()

def preprocess_image(image):
    image = tf.ensure_shape(image, [None, None, 3])  # Adjust shape to be flexible
    image = tf.image.resize(image, [256, 256])  # Resize to match model input shape
    image = image / 255.0  # Normalize the image to [0, 1]
    return image

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)

    # Decoder
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.concatenate([x, x])  # Skip connection
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.concatenate([x, x])  # Skip connection
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.concatenate([x, x])  # Skip connection
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(x)  # Assuming output is RGB image

    model = models.Model(inputs, outputs)
    return model

def build_deblur_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(conv5)

    # Decoder
    up1 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    concat1 = layers.concatenate([up1, conv4])  # Skip connection
    conv6 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(concat1)
    conv6 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv6)

    up2 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv6)
    concat2 = layers.concatenate([up2, conv3])  # Skip connection
    conv7 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(concat2)
    conv7 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv7)

    up3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv7)
    concat3 = layers.concatenate([up3, conv2])  # Skip connection
    conv8 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(concat3)
    conv8 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv8)

    up4 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv8)
    concat4 = layers.concatenate([up4, conv1])  # Skip connection
    conv9 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(concat4)
    conv9 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv9)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(conv9)  # Assuming output is RGB

    model = models.Model(inputs, outputs)
    return model

def get_image_paths(dataset_directory):
    data_dir = pathlib.Path(dataset_directory)
    all_image_paths = list(data_dir.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    return all_image_paths

def train_val_test_split(image_paths):
    train_paths, val_paths = train_test_split(image_paths, test_size=0.1)
    train_paths, test_paths = train_test_split(image_paths, test_size=1.0/9.0)
    return train_paths, val_paths, test_paths

def create_train_val_test_datasets(train_paths, val_paths, test_paths):
    clear_train = tf.data.Dataset.from_tensor_slices(train_paths)
    clear_val = tf.data.Dataset.from_tensor_slices(val_paths)
    clear_test = tf.data.Dataset.from_tensor_slices(test_paths)
    return clear_train, clear_val, clear_test

def load_train_val_test_images(clear_train, clear_val, clear_test):
    mapped_clear_train = clear_train.map(load_and_decode_image)
    mapped_clear_val = clear_val.map(load_and_decode_image)
    mapped_clear_test = clear_test.map(load_and_decode_image)
    return mapped_clear_train, mapped_clear_val, mapped_clear_test

def preprocess_train_val_test_images(clear_train, clear_val, clear_test):
    processed_clear_train = clear_train.map(lambda x: preprocess_image(x))
    processed_clear_val = clear_val.map(lambda x: preprocess_image(x))
    processed_clear_test = clear_test.map(lambda x: preprocess_image(x))
    return processed_clear_train, processed_clear_val, processed_clear_test

def blur_train_val_test_images(clear_train, clear_val, clear_test):
    blurred_train = clear_train.map(random_blur_image)
    blurred_val = clear_val.map(random_blur_image)
    blurred_test = clear_test.map(random_blur_image)
    return blurred_train, blurred_val, blurred_test

def display_sample_images(clear_train, blurred_train):
    clear_images = [image for image in clear_train.take(5)]
    blurred_images = [image for image in blurred_train.take(5)]
    show_images_side_by_side(clear_images, blurred_images, num_images=5)

def load_and_compile_model(input_shape, model_directory = None, output_summary = True, deeper_model = True):
    if model_directory:
        model = tf.keras.models.load_model(model_directory)
    else:
        if deeper_model:
            model = build_deblur_model(input_shape)
        else:
            model = build_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    if output_summary:
        model.summary()
    return model

def zip_train_val_test_images(blurred_train, clear_train, blurred_val, clear_val, blurred_test, clear_test):
    train_data = tf.data.Dataset.zip((blurred_train, clear_train))
    validation_data = tf.data.Dataset.zip((blurred_val, clear_val))
    test_data = tf.data.Dataset.zip((blurred_test, clear_test))
    return train_data, validation_data, test_data

def batch_train_val_test_images(train_data, validation_data, test_data):
    batched_train_data = train_data.shuffle(100).batch(8)
    batched_validation_data = validation_data.shuffle(100).batch(8)
    batched_test_data = test_data.shuffle(100).batch(8)
    return batched_train_data, batched_validation_data, batched_test_data

def display_train_val_loss(history):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def display_train_val_accuracy(history):
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

def display_training_plots(history):
    plt.figure(figsize=(12, 5))
    display_train_val_loss(history)
    display_train_val_accuracy(history)
    plt.show()

def get_test_batch(model, test_data):
    test_images, test_labels = next(iter(test_data))
    pred_images = model.predict(test_images)
    return test_images, test_labels, pred_images

def display_input_image(test_images, i):
    plt.subplot(3, len(test_images), i + 1)
    plt.imshow(test_images[i])
    plt.title("Blurred Image")
    plt.axis('off')

def display_actual_image(test_labels, i):
    plt.subplot(3, len(test_labels), i + 1 + len(test_labels))
    plt.imshow(test_labels[i])
    plt.title("Actual Image")
    plt.axis('off')

def display_predicted_image(pred_images, i):
    plt.subplot(3, len(pred_images), i + 1 + 2 * len(pred_images))
    plt.imshow(pred_images[i])
    plt.title("Predicted Image")
    plt.axis('off')

def display_predictions(test_images, test_labels, pred_images):
    plt.figure(figsize=(15, 7))
    for i in range(len(test_images)):
        display_input_image(test_images, i)
        display_actual_image(test_labels, i)
        display_predicted_image(pred_images, i)
    plt.show()



