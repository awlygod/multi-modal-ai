import torch.nn as nn
from transformers import BertModel
import torchvision.models as vision_models

# the code is defining a text encoder that uses BERT (a specific type of transformer model )
class TextEncoder(nn.Module):

    # This loads a pre-trained BERT model. It's like getting a model that has already learned
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    # This means we are using BERT as-is, without modifying its uncerstanding of the language
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
    
class  VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained = True)   
        for param in self.bert.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )