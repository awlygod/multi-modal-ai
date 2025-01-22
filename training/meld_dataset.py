from torch.utils.data import Dataset # base class for creating custom dataset in pytorch
import pandas as pd # for reading and handling csv data 
from transformers import AutoTokenizer # for converting text to BERT- format text
import os # for handling file paths 
import cv2 # for reading video files 
import numpy as np # for numerical operations 
import torch # the main pytorch library


class MELDDataset(Dataset): # it is like inheriting from Pytorch's Dataset class
    def __init__(self, csv_path, video_dir): 
        self.data = pd.read_csv(csv_path) # loads the csv containing metadata and labels
        self.video_dir = video_dir # directory containing video files 
        # initializing bert tokenizer for processing text
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Removed auth token for security
        # mapping dictionaries to convert emotions/sentiments to numerical labels
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }
    
    def _load_video_frames(self, video_path):
        # OpenCV obejct for reading values
        cap = cv2.VideoCapture(video_path)
        frames = []
        # Error handling and video reading 
        try:
            # Check if video exists and can be opened 
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video 
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Resizing the frame to 224*224 and normalize pixel values to [0,1]
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
            
        except Exception as e:
            raise ValueError(f"Video not found: {video_path}")
        finally: 
            cap.release()
            
        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        # Handle cases where we get fewer than 30 frames 
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [Frames, Height, Width, Channels]
        # After permute: [Frames, Channels, Height, Width]
        # conver to pytorch
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def __len__(self):
        # return total no of samples in dataset 
        return len(self.data)
    
    def __getitem__(self, index):
        # get one same of data 
        row = self.data.iloc[index]
        # Construct video filename
        video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        path = os.path.join(self.video_dir, video_filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No video file found for filename: {path}")

        # Get text inputs and squeeze to remove batch dimension / processing text using BERT Tokenizer
        text_inputs = self.tokenizer(
            row['Utterance'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        # Get video frames
        video_frames = self._load_video_frames(path)
        print (video_frames)
        
        # Return all componenets as a dictionary
        return {
            'text_inputs': text_inputs,
            'video_frames': video_frames,
        }


if __name__ == "__main__":
    meld = MELDDataset('../dataset/dev/dev_sent_emo.csv',
                       '../dataset/dev/dev_splits_complete')
    
    sample = meld[0]
    print (meld[0])