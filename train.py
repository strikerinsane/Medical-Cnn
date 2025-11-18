"""
Training Module
Handles model training, validation, and performance evaluation.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_loader import create_data_generators
from model import create_cnn_model, save_model

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history (loss and accuracy).

    Args:
        history: Keras History object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Bleeding'],
                yticklabels=['Normal', 'Bleeding'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def evaluate_model(model, x_val, y_val):
    """
    Evaluate model on validation data.

    Args:
        model: Trained Keras model
        x_val: Validation images
        y_val: Validation labels
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION ON VALIDATION DATA")
    print("="*80)

    # Get predictions
    y_pred_proba = model.predict(x_val, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    loss, accuracy, precision, recall, auc = model.evaluate(x_val, y_val, verbose=0)

    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation AUC: {auc:.4f}")

    # Classification report
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT")
    print("-"*80)
    print(classification_report(y_val, y_pred, 
                                target_names=['Normal', 'Bleeding'],
                                digits=4))

    # Confusion matrix
    print("-"*80)
    print("CONFUSION MATRIX")
    print("-"*80)
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print("-"*80)

    # Plot confusion matrix
    plot_confusion_matrix(y_val, y_pred)

    print("="*80 + "\n")

    return y_pred

def train_model(args):
    """
    Main training function.

    Args:
        args: Command-line arguments
    """
    print("\n" + "="*80)
    print("MEDICAL IMAGE CLASSIFICATION - INTERNAL BLEEDING DETECTION")
    print("="*80 + "\n")

    # Create data loader
    print("Initializing data loader...")
    data_loader = create_data_generators(img_size=(args.img_size, args.img_size))

    # Load training data
    print(f"\nLoading training data from {args.training_dir}...")
    x_train, y_train = data_loader.load_training_data(
        training_dir=args.training_dir,
        apply_augmentation=True
    )

    # Load validation data
    print(f"\nLoading validation data from {args.validation_dir}...")
    x_val, y_val = data_loader.load_validation_data(
        validation_dir=args.validation_dir
    )

    # Calculate class weights
    class_weights = data_loader.get_class_weights(y_train)

    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model(
        input_shape=(args.img_size, args.img_size, 3),
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            args.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80 + "\n")

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    evaluate_model(model, x_val, y_val)

    # Save final model
    save_model(model, args.model_path)

    print("\nTraining completed successfully!")
    print(f"Model saved to: {args.model_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train CNN model for medical image classification'
    )

    parser.add_argument('--training_dir', type=str, default='./training',
                       help='Path to training data directory')
    parser.add_argument('--validation_dir', type=str, default='./validation',
                       help='Path to validation data directory')
    parser.add_argument('--model_path', type=str, default='medical_cnn_model.h5',
                       help='Path to save trained model')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size (height and width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')

    args = parser.parse_args()

    # Train model
    train_model(args)

if __name__ == '__main__':
    main()
