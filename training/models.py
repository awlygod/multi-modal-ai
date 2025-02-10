import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as vision_models
from meld_dataset import MELDDataset

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
            nn.Conv1d(64,64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False
        
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

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
        )
        # Classification heads 
        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,7) # Sadness , anger 
        )

        self.sentiment_classfier = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,3) # Negative , positive , neutral

        )

    def forward(self, text_inputs, video_frames, audio_features) :
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate multimodal features 
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1) 

        fused_features = self.fusion_layer(combined_features)
        emotion_output = self.emo_classifier(fused_features)
        sentiment_output = self.sentiment_classfier(fused_features)

        return {
            'emotions' : emotion_output,
            'sentiments': sentiment_output
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

