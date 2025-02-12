import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as vision_models
from meld_dataset import MELDDataset
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
# Thinking of this code as a system that understands emotions and sentiments from videos
# What they are saying (text)
# How they look (video)
# How they sound (audio)

# the code is defining a text encoder that uses BERT (a specific type of transformer model )
class TextEncoder(nn.Module):

    # This loads a pre-trained BERT model. It's like getting a model that has already learned
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    # This means we are using BERT as-is, without modifying its understanding of the language
        for param in self.bert.parameters():
            param.requires_grad = False
    # This reduces BERT's 768 dimensional output to 128 dimensions, making it more compact      
        self.projection = nn.Linear(768, 128)

    # input_ids => these are numbers that represent your text basically tokenization
    #              BERT converts each word into a number
    # attention-mask => list of 0s and 1s. 1 means "pay attention to this word" and 0 means "ignore this" 
    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings ie BERT reads the text and creates rich represenations for each word
        # this captures the meaning and context of each word in your text
        outputs = self.bert(input_ids = input_ids, attention_mask= attention_mask)

        # Use [CLS] token representation 
        pooler_output = outputs.pooler_output
        return self.projection(pooler_output)

# This creates a new neural network module that inherits from nn.module(nn.module is basically a base class for all neural network modules)    
class  VideoEncoder(nn.Module):
    # Initializes the class and calls the parent class's constructor
    def __init__(self):
        super().__init__() 
        # Creates the main architecture using a pre trained R3D-18 model
        # R3D-18 is 3d ResNet model with 18 layers, designed specifically for video processing 
        self.backbone = vision_models.video.r3d_18(pretrained = True) 
        # This is written to freeze the parameters of the backbone model soo they don't get updated during training
        for param in self.backbone.parameters():
            param.requires_grad = False
        # This line is checking how much information is coming in. Imagine you have a video that contains 512 pieces of info about each scene
        num_fts = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            # This is like having a compressor that takes your 512 pieces of information and cleverly combines them into just 128 pieces 
            nn.Linear(num_fts, 128),
            # Implementing this would mean if a piece of info is useful (positive), keep it
            nn.ReLU(),
            # This is like randomly ignoring 20% of the information during training 
            nn.Dropout(0.2)
        )
    def forward(self, x):
        # [batch_size , frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1,2) 
        return self.backbone(x)  

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(

            # Lower level features
            nn.Conv1d(64,64, kernel_size=3), # takes in 64 channels of audio, kernel looks at 3 time steps at once 
            nn.BatchNorm1d(64), # Normalizes the data 
            nn.ReLU(), # Activation fucntion
            nn.MaxPool1d(2), # Reduces the time dimension by half 

            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3), # increases the channel to 128 for more features
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Averages all time dimension by half
        )
        # Freeze the weights- they won't be updated during training 
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        # Final processing of audio features
        self.projection = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))

class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders 
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion Layer : Combines all features together
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256), # Takes 384 inputs (128 from each encoder)
            nn.BatchNorm1d(256), # Normalizes the combined features
            nn.ReLU(), # Activation function 
            nn.Dropout(0.3), # Prevents overfitting
            
        )
        # Emotion classifier (predicts 7 emotions)
        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,7) # Outputs : 7 emotion scores
        )
        # Sentiment classifier (predicts positive/negative/neutral)
        self.sentiment_classfier = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,3) # Outputs 3 sentiment scores

        )

    def forward(self, text_inputs, video_frames, audio_features) :
        # Get features from each encoder
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Combine all features
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1) 
        # Process combined features
        fused_features = self.fusion_layer(combined_features)
        # Make predictions    
        emotion_output = self.emo_classifier(fused_features)
        sentiment_output = self.sentiment_classfier(fused_features)

        return {
            'emotions' : emotion_output,
            'sentiments': sentiment_output
        }



class MultimodalTrainer:
            def __init__(self,model,train_loader, val_loader):
                # Store the model and data loaders as class attribtues
                self.model = model          # The multimodal sentiment model
                self.train_loader = train_loader   # Dataloader for training data
                self.val_loader = val_loader  # Dataloader for validation data

                # Print dataset information
                train_size = len(train_loader.dataset)  # Get total training samples
                val_size = len(val_loader.dataset) # Get total validation samples
                print("\nDataset sizes:")
                print(f"Training samples: {train_size:,}")
                print(f"Validation samples: {val_size:,}")
                print(f"Batches per epoch: {len(train_loader):,}")

                # Setup Tensorboard logging
                timestamp = datetime.now().strftime('%b%d_%H-%M-%S')  # Create timestamp 
                # Choose directory based on environment (Sagemaker or local )
                base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
                log_dir = f"{base_dir}/run_{timestamp}"
                self.writer = SummaryWriter(log_dir=log_dir)  # Initialize Tensorboard writer
                self.global_step = 0    # Counter for training steps

                # Set up optimizer with different learning rates for different parts
                # Very high :1 , high: 0.1-0.01, medium: 1e-1, low: 1e-4, very low: 1e-5
                self.optimizer = torch.optim.Adam([
                    {'params': model.text_encoder.paramters(), 'lr': 8e-6},  # Very low learning rate
                    {'params': model.video_encoder.paramters(), 'lr': 8e-5}, # Higher learning rate
                    {'params': model.audio_encoder.paramters(), 'lr': 8e-5}, # Higher learning rate
                    {'params': model.fusion_layer.paramters(), 'lr': 5e-4},  # Highest learning rate
                    {'params': model.emotion_classifier.paramters(), 'lr': 5e-6},  # Very low learning rate
                    {'params': model.sentiment_classifier.paramters(), 'lr': 5e-4} # Highest learning rate
                
                ], weight_decay = 1e-5)     # Add weight decay to prevent overfitting

                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode = "min",   # Reduce LR when monitored value stops decreasing
                    factor = 0.1,   # Multiply LR by 0.1 when reducing
                    patience= 2     # Wait 2 epochs before reducing LR
                )

                # Loss functions for both tasks
                self.emotion_criterion = nn.CrossEntropyLoss(
                    label_smoothing=0.05    # Add label smoothing to prevent overconfident predictions
                )
                self.sentiment_criterion = nn.CrossEntropyLoss(
                    label_smoothing=0.05
                )
            def log_metrics(self, losses, metrics=None, phase="train"):
                # For training phase, store current losses
                if phase == "train":
                    self.current_train_losses = losses
                else:  # For validation phase, log both train and val losses
                    # Log total loss
                    self.writer.add_scalar(
                        'loss/total/train', self.current_train_losses['total'], self.global_step)
                    self.writer.add_scalar(
                        'loss/total/val', losses['total'], self.global_step)
                    # Log emotion loss
                    self.writer.add_scalar(
                        'loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
                    self.writer.add_scalar(
                        'loss/emotion/val', losses['emotion'], self.global_step)
                    # Log sentiment loss
                    self.writer.add_scalar(
                        'loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
                    self.writer.add_scalar(
                        'loss/sentiment/val', losses['sentiment'], self.global_step)
                # If metrics are provided, log them
                if metrics:
                    self.writer.add_scalar(
                        f'{phase}/emotion_precision', metrics['emotion_precision'], self.global_step)
                    self.writer.add_scalar(
                        f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step)
                    self.writer.add_scalar(
                        f'{phase}/sentiment_precision', metrics['sentiment_precision'], self.global_step)
                    self.writer.add_scalar(
                        f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'], self.global_step)  

            def train_epoch(self):
                # Set model to training mode (enables dropout, batch norm updates, etc.)
                self.model.train()
                # Initialize running losses for tracking
                running_loss = {'total':0, 'emotion':0, 'sentiment': 0}
                # Loop through each batch in training data
                for batch in self.train_loader:
                    # Get the device (CPU/GPU) where model is located
                    device = next(self.model.parameters()).device
                    # Prepare text inputs and move to device
                    text_inputs = {
                        'input_ids': batch['text_input']['input_ids'].to(device),
                        'attention_mask': batch['text_inpt']['attention_mask'].to(device)
                    }
                    # Move other inputs to device
                    video_frames = batch['video_frames'].to(device)
                    audio_features = batch['audio_frames'].to(device)
                    emotion_labels = batch['emotion_label'].to(device)
                    sentiment_labels = batch['sentiment_label'].to(device)

                    # Clear gradients from previous batch
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(text_inputs, video_frames, audio_features)

                    # Calculate losses using raw logics
                    emotion_loss = self.emotion_criterion(
                        outputs["emotion"], emotion_labels)
                    sentiment_loss = self.sentiment_criterion(
                        outputs["emotion"], sentiment_labels)
                    total_loss = emotion_loss + sentiment_loss

                    # Backward pass. Calculate gradients
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm =1.0)

                    self.optimizer.step()

                    # Track losses 
                    running_loss['total'] += total_loss.item()
                    running_loss['emotion'] += emotion_loss.item()
                    running_loss['sentiment'] += sentiment_loss.item()

                    self.log_metrics({
                        'total': total_loss.item(),
                        'emotion': emotion_loss.item(),
                        'sentiment': sentiment_loss.item()
                    })
                    self.global_step += 1

                return {k: v/len(self.train_loader) for k,v in running_loss.items()}


            def evaluate(self, data_loader, phase= "val"):
                self.model.eval()
                losses = { 'total': 0, 'emotion': 0, 'sentiment': 0}
                all_emotion_preds = []
                all_emotion_labels = []
                all_sentiment_preds = []
                all_sentiment_labels = []

                with torch.inference_model():
                    for batch in data_loader:
                        device = next(self.model.oarameters()).device
                        text_inputs = {
                            'input_ids': batch['text_inputs']['input_ids'].to(device),
                            'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                        }
                        video_frames = batch['video_frames'].to(device)
                        audio_features = batch['audio_frames'].to(device)
                        emotion_labels = batch['emotion_label'].to(device)
                        sentiment_labels = batch['sentiment_label'].to(device)

                        outputs = self.model(text_inputs, video_frames, audio_features)

                        emotion_loss = self.emotion_criterion(
                            outputs["emotions"], emotion_labels)
                        sentiment_loss = self.sentiment_criterion(
                            outputs["sentiments"], sentiment_labels)
                        total_loss = emotion_loss + sentiment_loss
                        
                        all_emotion_preds.extend(
                            outputs["emotions"].argmax(dim=1).cpu().numpy())
                        all_emotion_labels.extend(emotion_labels.cpu().numpy())
                        all_sentiment_preds.extend(
                            outputs["sentiments"].argmax(dim=1).cpu().numpy())
                        all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                        losses['total'] += total_loss.item()
                        losses['emotion'] += emotion_loss.item()
                        losses['sentiment'] += sentiment_loss.item()
                

                avg_loss = {k: v/len(data_loader) for k, v in losses.items()} 

                 # Compute the precision and accuracy
                emotion_precision = precision_score(
                    all_emotion_labels, all_emotion_preds, average='weighted')
                emotion_accuracy = accuracy_score(
                    all_emotion_labels, all_emotion_preds)
                sentiment_precision = precision_score(
                    all_sentiment_labels, all_sentiment_preds, average='weighted')
                sentiment_accuracy = accuracy_score(
                    all_sentiment_labels, all_sentiment_preds)

                self.log_metrics(avg_loss, {
                    'emotion_precision': emotion_precision,
                    'emotion_accuracy': emotion_accuracy,
                    'sentiment_precision': sentiment_precision,
                    'sentiment_accuracy': sentiment_accuracy
                }, phase=phase)

                if phase == "val":
                    self.scheduler.step(avg_loss['total'])

                return avg_loss, {
                    'emotion_precision': emotion_precision,
                    'emotion_accuracy': emotion_accuracy,
                    'sentiment_precision': sentiment_precision,
                    'sentiment_accuracy': sentiment_accuracy
                    }


if __name__ == "__main__": 
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits'
    )
    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()
    text_inputs = {
        'input_ids' : sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_frames'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        emotion_probs = torch.softmax(outputs['emotions'], dim =1) [0]
        sentiment_probs = torch.softmax(outputs['sentiments'],dim=1) [0]
    
    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4:'neutral', 5: 'sadness', 6: 'surprise'
    }
    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }
    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    for i,prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

