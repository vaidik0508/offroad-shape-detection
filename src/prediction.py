import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, List, Tuple

from .model import LaneDetectionModel
from .curve_utils import plot_curve_using_control_points

class LanePredictor:
    """Class for making predictions on images and videos using a trained lane detection model."""
    
    def __init__(self, model_path: str, input_shape: Tuple[int, int, int] = (480, 640, 3)):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model weights (.h5 file)
            input_shape (tuple): Expected input shape (height, width, channels)
        """
        self.model = LaneDetectionModel(input_shape)
        self.model.load_weights(model_path)
        self.input_shape = input_shape
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Convert to float32 and normalize
        image = image.astype(np.float32) / 255.0
        return image
    
    def predict_on_image(self, image: np.ndarray, visualize: bool = True) -> Union[np.ndarray, List[Tuple[float, float]]]:
        """
        Predict lane boundaries on a single image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            visualize (bool): Whether to draw the predicted lane on the image
            
        Returns:
            Union[np.ndarray, List[Tuple[float, float]]]: 
                If visualize=True: Image with predicted lane drawn
                If visualize=False: List of control points
        """
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        # Make prediction
        control_points = self.model.predict(processed_img)
        
        # Reshape control points to list of tuples
        control_points = [(control_points[i], control_points[i+1]) 
                         for i in range(0, len(control_points), 2)]
        
        if visualize:
            # Create a copy of the image for visualization
            vis_img = image.copy()
            # Plot curve using control points
            fig = plot_curve_using_control_points(vis_img, control_points)
            # Convert matplotlib figure to OpenCV image
            fig.canvas.draw()
            plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            return plot_img
        
        return control_points
    
    def process_video(self, input_path: str, output_path: str, show_progress: bool = True):
        """
        Process a video file and save the output with predicted lanes.
        
        Args:
            input_path (str): Path to input video file
            output_path (str): Path to save output video
            show_progress (bool): Whether to show processing progress
        """
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.predict_on_image(frame)
            
            # Write frame
            out.write(processed_frame)
            
            # Show progress
            if show_progress:
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                print(f"\rProcessing: {progress:.1f}%", end="")
        
        # Clean up
        cap.release()
        out.release()
        if show_progress:
            print("\nVideo processing completed!")

def main():
    """Example usage of the LanePredictor class."""
    # Initialize predictor
    predictor = LanePredictor("models/final_model.h5")
    
    # Example: Process a single image
    image_path = "test_images/test1.jpg"
    if Path(image_path).exists():
        img = cv2.imread(image_path)
        result = predictor.predict_on_image(img)
        cv2.imwrite("test_images/result1.jpg", result)
    
    # Example: Process a video
    video_path = "test_videos/test1.mp4"
    if Path(video_path).exists():
        predictor.process_video(
            video_path,
            "test_videos/result1.mp4"
        )

if __name__ == "__main__":
    main()

