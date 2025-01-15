import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

class LaneDetectionModel:
    """Neural network model for offroad lane detection."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (480, 640, 3)):
        """
        Initialize the lane detection model.
        
        Args:
            input_shape (Tuple[int, int, int]): Input image shape (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self) -> models.Model:
        """
        Build the neural network architecture.
        
        Returns:
            models.Model: Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Feature processing
        x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Global features
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer - 8 values for 4 control points (x,y)
        outputs = layers.Dense(8)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, train_data: tf.data.Dataset, validation_data: tf.data.Dataset, 
              epochs: int = 100, callbacks: list = None) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_data (tf.data.Dataset): Training dataset
            validation_data (tf.data.Dataset): Validation dataset
            epochs (int): Number of training epochs
            callbacks (list): List of Keras callbacks
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks
        )
    
    def predict(self, image: tf.Tensor) -> tf.Tensor:
        """
        Predict lane control points for an input image.
        
        Args:
            image (tf.Tensor): Input image tensor
            
        Returns:
            tf.Tensor: Predicted control points
        """
        return self.model.predict(tf.expand_dims(image, 0))[0]
    
    def save_weights(self, filepath: str):
        """Save model weights to a file."""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Load model weights from a file."""
        self.model.load_weights(filepath) 