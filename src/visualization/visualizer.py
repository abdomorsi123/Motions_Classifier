import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging
from glob import glob
import torch
import random


logger = logging.getLogger(__name__)

class SkeletonVisualizer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualizer with configuration
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config
        self.fps = config['visualization']['fps']
        self.figure_size = config['visualization']['figure_size']
        self.colors = config['visualization']['keypoint_colors']
        
        # Define keypoint connections for skeleton visualization
        self.keypoint_connections = [
            (0, 1),   # Nose to Neck
            (1, 2),   # Neck to RShoulder
            (2, 3),   # RShoulder to RElbow
            (3, 4),   # RElbow to RWrist
            
            (1, 5),   # Neck to LShoulder
            (5, 6),   # LShoulder to LElbow
            (6, 7),   # LElbow to LWrist
            
            (1, 8),   # Neck to MidHip
            (8, 9),   # MidHip to RHip
            (8, 12),  # MidHip to LHip
            
            (0, 15),  # Nose to REye
            (0, 16),  # Nose to LEye
            
            (15, 17), # REye to REar
            (16, 18)  # LEye to LEar
        ]
        
        # Create output directories
        self.frames_dir = Path(config['paths']['frames_dir'])
        self.outputs_dir = Path(config['paths']['outputs_dir'])
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_connection_color(self, start_idx: int, end_idx: int) -> str:
        """
        Determine the color for a skeleton connection
        
        Args:
            start_idx: Starting keypoint index
            end_idx: Ending keypoint index
            
        Returns:
            Color string for the connection
        """
        if start_idx in [2,3,4] or end_idx in [2,3,4]:
            return self.colors['right_arm']
        elif start_idx in [5,6,7] or end_idx in [5,6,7]:
            return self.colors['left_arm']
        elif start_idx in [0,1,8] or end_idx in [0,1,8]:
            return self.colors['spine']
        else:
            return self.colors['face']
    
    def _create_index_mapping(self) -> Dict[int, int]:
        """
        Create mapping from connection indices to data indices
        
        Returns:
            Dictionary mapping skeleton indices to data indices
        """
        return {
            0: 0,    # Nose
            1: 3,    # Neck
            2: 6,    # RShoulder
            3: 9,    # RElbow
            4: 12,   # RWrist
            5: 15,   # LShoulder
            6: 18,   # LElbow
            7: 21,   # LWrist
            8: 24,   # MidHip
            9: 27,   # RHip
            12: 30,  # LHip
            15: 33,  # REye
            16: 36,  # LEye
            17: 39,  # REar
            18: 42   # LEar
        }
    
    def visualize_frame(
        self,
        frame_data: pd.Series,
        frame_num: int,
        subject: int,
        action: str,
        iteration: int
    ) -> str:
        """
        Visualize a single frame of skeleton data
        
        Args:
            frame_data: Series containing the frame data
            frame_num: Frame number
            subject: Subject ID
            action: Action name
            iteration: Iteration number
            
        Returns:
            Path to the saved frame image
        """
        plt.figure(figsize=self.figure_size)
        plt.title(f"Subject {subject} - {action} - {iteration} - Frame {frame_num}")
        
        # Set the aspect ratio to match the original video dimensions (640x480)
        plt.gca().set_aspect(480/640)
        
        # Get index mapping
        index_mapping = self._create_index_mapping()
        
        # Plot keypoints and connections
        for connection in self.keypoint_connections:
            try:
                start_idx = index_mapping[connection[0]]
                end_idx = index_mapping[connection[1]]
                
                x1, y1 = frame_data[start_idx], frame_data[start_idx + 1]
                x2, y2 = frame_data[end_idx], frame_data[end_idx + 1]
                
                color = self._get_connection_color(connection[0], connection[1])
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2)
                
            except (KeyError, IndexError) as e:
                logger.warning(f"Could not plot connection {connection}. Error: {str(e)}")
                continue
        
        plt.xlim(-50, 700)
        plt.ylim(500, -50)
        
        # Create frame directory and save the plot
        frame_dir = self.frames_dir / f"p{subject}_{action}_{iteration}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        frame_path = frame_dir / f"frame_{frame_num:03d}.png"
        plt.savefig(frame_path)
        plt.close()
        
        return str(frame_path)
    
    def create_video(
        self,
        subject: int,
        action: str,
        iteration: int,
        frame_pattern: str = "frame_*.png"
    ) -> str:
        """
        Create a video from a sequence of frames
        
        Args:
            subject: Subject ID
            action: Action name
            iteration: Iteration number
            frame_pattern: Pattern to match frame files
            
        Returns:
            Path to the created video file
        """
        # Get frame directory and pattern
        frame_dir = self.frames_dir / f"p{subject}_{action}_{iteration}"
        frames = sorted(glob(str(frame_dir / frame_pattern)))
        
        if not frames:
            raise ValueError(f"No frames found in {frame_dir}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frames[0])
        height, width, layers = first_frame.shape
        
        # Create output video file
        output_file = self.outputs_dir / f"p{subject}_{action}_{iteration}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, self.fps, (width, height))
        
        # Write frames to video
        for frame_path in frames:
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved to {output_file}")
        
        return str(output_file)
    
    def visualize_sequence(
        self,
        sequence: pd.DataFrame,
        subject: int,
        action: str,
        iteration: int
    ) -> str:
        """
        Visualize an entire sequence of frames and create a video
        
        Args:
            sequence: DataFrame containing the sequence data
            subject: Subject ID
            action: Action name
            iteration: Iteration number
            
        Returns:
            Path to the created video file
        """
        logger.info(f"Visualizing sequence: Subject {subject}, Action {action}, Iteration {iteration}")
        
        # Generate frames
        for frame in range(len(sequence)):
            self.visualize_frame(sequence.iloc[frame], frame, subject, action, iteration)
        
        # Create video
        return self.create_video(subject, action, iteration)
    
    # Function to load and visualize a specific sequence
    def visualize_specific_sequence(self, subject: int, action: str, iteration: int):
        """
        Visualize a specific sequence from a CSV file
        
        Args:
            subject: Subject ID
            action: Action name
            iteration: Iteration number
        """
        # Format iteration to be two digits
        iteration_str = f"{iteration:02d}"

        # Construct the file name dynamically
        csv_file_name = f"p{subject}_{action}_{iteration_str}.csv"

        csv_path = os.path.join(
            self.config['data']['base_path'], 
            self.config['data']['train_dir'], 
            csv_file_name
        )
        
        # Load the sequence directly from CSV
        sequence_df = pd.read_csv(csv_path, header=None)
        
        # Create visualization
        video_path = self.visualize_sequence(
            sequence_df,
            subject=subject,
            action=action,
            iteration=iteration
        )
        return video_path

    # Function to load and visualize a random sequence
    def visualize_random_sequence(self):
        """
        Visualize a specific sequence with random arguments.
        """
        actions = ["boxing", "drums", "guitar", "rowing", "violin"]

        # Select random subject, action, and iteration
        subject = random.randint(1, 30)
        action = random.choice(actions)
        iteration = random.randint(1, 10)

        # Format iteration to be two digits
        iteration_str = f"{iteration:02d}"

        # Construct the file name dynamically
        csv_file_name = f"p{subject}_{action}_{iteration_str}.csv"

        csv_path = os.path.join(
            self.config['data']['base_path'], 
            self.config['data']['train_dir'], 
            csv_file_name
        )

        try:
            # Load the sequence directly from CSV
            sequence_df = pd.read_csv(csv_path, header=None)

            # Create visualization
            video_path = self.visualize_sequence(
                sequence_df,
                subject=subject,
                action=action,
                iteration=iteration
            )
            return video_path

        except FileNotFoundError:
            print(f"File not found: {csv_path}")
            return None