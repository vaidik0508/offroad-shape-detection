import os
import tensorflow as tf
from datetime import datetime
from pathlib import Path

from data_loader import OffroadDataLoader
from model import LaneDetectionModel

def create_tf_dataset(data_loader: OffroadDataLoader, batch_size: int, shuffle: bool = True):
    """Create a TensorFlow dataset from the data loader."""
    def generator():
        for images, control_points in data_loader.get_batch(batch_size, shuffle):
            yield tf.convert_to_tensor(images), tf.convert_to_tensor(control_points)
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4, 2), dtype=tf.float32)
        )
    )

def main():
    # Configuration
    IMG_DIR = "raw_data/final_img"
    LABEL_FILE = "raw_data/bezier.csv"
    BATCH_SIZE = 32
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    
    # Create data loader
    data_loader = OffroadDataLoader(IMG_DIR, LABEL_FILE)
    dataset_size = len(data_loader)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    # Create TensorFlow datasets
    train_loader = OffroadDataLoader(
        IMG_DIR, 
        LABEL_FILE
    )
    val_loader = OffroadDataLoader(
        IMG_DIR, 
        LABEL_FILE
    )
    
    train_dataset = create_tf_dataset(train_loader, BATCH_SIZE)
    val_dataset = create_tf_dataset(val_loader, BATCH_SIZE, shuffle=False)
    
    # Create and compile model
    model = LaneDetectionModel()
    
    # Setup callbacks
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/model_{epoch:02d}.h5',
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.train(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save final model
    model.save_weights('models/final_model.h5')

if __name__ == "__main__":
    main() 