import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List
from pathlib import Path

class OffroadDataLoader:
    """Data loader class for offroad lane detection dataset."""
    
    def __init__(self, img_dir: str, label_file: str):
        """
        Initialize the data loader.
        
        Args:
            img_dir (str): Directory containing the images
            label_file (str): Path to the CSV file containing bezier curve control points
        """
        self.img_dir = Path(img_dir)
        self.labels_df = pd.read_csv(label_file, index_col=0)
        
    def load_image(self, img_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        img_full_path = self.img_dir / img_path
        img = cv2.imread(str(img_full_path))
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_full_path}")
        return img
    
    def get_control_points(self, img_path: str) -> List[Tuple[float, float]]:
        """
        Get bezier curve control points for an image.
        
        Args:
            img_path (str): Image file name
            
        Returns:
            List[Tuple[float, float]]: List of control points as (x,y) coordinates
        """
        row = self.labels_df[self.labels_df['path'] == img_path].iloc[0]
        control_points = []
        for i in range(1, 5):
            control_points.append((row[f'x_{i}'], row[f'y_{i}']))
        return control_points
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.labels_df)
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[List[np.ndarray], List[List[Tuple[float, float]]]]:
        """
        Get a batch of images and their corresponding control points.
        
        Args:
            batch_size (int): Number of images to load
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            Tuple[List[np.ndarray], List[List[Tuple[float, float]]]]: 
                Batch of images and their control points
        """
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(self), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_paths = self.labels_df.iloc[batch_indices]['path'].values
            
            images = []
            control_points = []
            
            for path in batch_paths:
                images.append(self.load_image(path))
                control_points.append(self.get_control_points(path))
            
            yield images, control_points 