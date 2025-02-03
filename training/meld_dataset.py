from torch.utils.data import Dataset, DataLoader # base class for creating custom dataset in pytorch
import pandas as pd # for reading and handling csv data 
import torch.utils.data.dataloader
from transformers import AutoTokenizer # for converting text to BERT- format text
import os # for handling file paths 
import cv2 # for reading video files 
import numpy as np # for numerical operations 
import torch # the main pytorch library
import subprocess
import torchaudio

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
    def _extract_audio_features(self, video_path):
            audio_path = video_path.replace('.mp4', '.wav')

            try:
                subprocess.run([
                    'ffmpeg',
                    '-y', # adding this flag to force overwrite
                    '-i',video_path,
                    '-vn',
                    '-acodec','pcm_s16le',
                    '-ar', '16000',
                    '-ac','1',
                    audio_path
                ], check =True,  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                waveform,sample_rate = torchaudio.load(audio_path)

                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)


                mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_mels=64,
                    n_fft=1024,
                    hop_length=512
                )

                mel_spec = mel_spectrogram(waveform)

                # Normalizing 
                mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

                if mel_spec.size(2) <300:
                    padding = 300 - mel_spec.size(2)
                    mel_spec = torch.nn.functional.pad(mel_spec, (0,padding))
                else:
                    mel_spec = mel_spec[:, :, :300]
                
                return mel_spec

            except subprocess.CalledProcessError as e:
                raise ValueError(f"Audio extraxction error: {str(e)}")
            except Exception as e:
                raise ValueError(f"Audio error: {str(e)}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

    def __len__(self):
        # return total no of samples in dataset 
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            idx = idx.item()
        # get one same of data 
        row = self.data.iloc[index]
        # Construct video filename
        try:
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
            #print (video_frames)
            
            audio_features = self._extract_audio_features(path)
            # Map sentiment and emotion labels
 
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label= self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs' : {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_frames': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return None

        # Return all componenets as a dictionary
        return {
            'text_inputs': text_inputs,
            #'video_frames': video_frames,
        }
def collate_fn(batch):
    # Filter out None samples
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size = 32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset,
                              batch_size= batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
                             # num_workers=4,
                             # pin_memory=True )
    dev_loader = DataLoader(dev_dataset,batch_size=batch_size,collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader
if __name__ == "__main__":
    train_loader , dev_loader, test_loader = prepare_dataloaders (
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )   


    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_frames'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break

    


