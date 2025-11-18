"""
CNN Model Architecture Module
Defines the CNN model built from scratch for binary classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def build_cnn_model(input_shape=(128, 128, 3), dropout_rate=0.5):
    """
    Build CNN model from scratch for medical image classification.

    Args:
        input_shape: Shape of input images (height, width, channels)
        dropout_rate: Dropout rate for regularization
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     kernel_regularizer=l2(0.001), name='conv1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001), name='conv1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.BatchNormalization(name='bn1'),
        layers.Dropout(dropout_rate * 0.5, name='dropout1'),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001), name='conv2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001), name='conv2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.BatchNormalization(name='bn2'),
        layers.Dropout(dropout_rate * 0.6, name='dropout2'),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001), name='conv3_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001), name='conv3_2'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.BatchNormalization(name='bn3'),
        layers.Dropout(dropout_rate * 0.7, name='dropout3'),

        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001), name='conv4_1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001), name='conv4_2'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        layers.BatchNormalization(name='bn4'),
        layers.Dropout(dropout_rate * 0.8, name='dropout4'),

        # Flatten and dense layers
        layers.Flatten(name='flatten'),

        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001), name='dense1'),
        layers.BatchNormalization(name='bn_dense1'),
        layers.Dropout(dropout_rate, name='dropout_dense1'),

        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense2'),
        layers.BatchNormalization(name='bn_dense2'),
        layers.Dropout(dropout_rate, name='dropout_dense2'),

        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense3'),
        layers.Dropout(dropout_rate * 0.5, name='dropout_dense3'),

        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output')
    ])

    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the CNN model with optimizer and loss function.

    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')]
    )

    return model

def get_model_summary(model):
    """
    Print model architecture summary.

    Args:
        model: Keras model
    """
    print("\n" + "="*80)
    print("CNN MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80 + "\n")

    # Count trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Total trainable parameters: {trainable_params:,}")
    print()

def create_cnn_model(input_shape=(128, 128, 3), learning_rate=0.001, dropout_rate=0.5):
    """
    Create and compile CNN model.

    Args:
        input_shape: Shape of input images
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout rate for regularization
    Returns:
        Compiled Keras model
    """
    model = build_cnn_model(input_shape=input_shape, dropout_rate=dropout_rate)
    model = compile_model(model, learning_rate=learning_rate)
    get_model_summary(model)
    return model

def save_model(model, model_path='medical_cnn_model.h5'):
    """
    Save trained model to file.

    Args:
        model: Trained Keras model
        model_path: Path to save model
    """
    model.save(model_path)
    print(f"Model saved to: {model_path}")

def load_saved_model(model_path='medical_cnn_model.h5'):
    """
    Load saved model from file.

    Args:
        model_path: Path to saved model
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model
