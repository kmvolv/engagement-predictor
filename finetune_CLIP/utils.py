import torch.nn as nn
from PIL import Image
import torch
from transformers import CLIPModel, BertModel
from torchvision import models
from torch.utils.data import Dataset
from alive_progress import alive_bar

NUM_ENGAGEMENT_METRICS = 3
new_cache_dir = "/work/classtmp/dhawal04/.cache/torch"
torch.hub.set_dir(new_cache_dir)

class PixelRecDataset(Dataset):
    def __init__(self, data, image_transform, tokenizer, normalizer):
        self.data = data
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.normalizer = normalizer
       
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        item = self.data[idx]
       
        # Load and transform image
        image = Image.open(item['image']).convert('RGB')
        image = self.image_transform(image)
       
        # Combine text fields
        text = f"Title: {item['title']} Tag: {item['tag']} Description: {item['description']}"
        text_encoding = self.tokenizer(text, padding='max_length',
                                       truncation=True, max_length=128,
                                       return_tensors='pt')
       
        # Use normalizer to normalize engagement metrics
        normalized_engagement = self.normalizer.transform(item['engagement'])
        # normalized_engagement is now a numpy array: [likes, comments, shares, views, favorites]
       
        engagement = torch.tensor([
            normalized_engagement[0],  # likes
            normalized_engagement[1],  # comments
            # normalized_engagement[2],  # shares
            normalized_engagement[2],  # views, actually position 3!!
            # normalized_engagement[4]   # favorites
        ], dtype=torch.float32)
       
        return {
            'image': image,
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'engagement': engagement
        }


class CLIPBasedEngagementPredictor(nn.Module):
    """
    Alternative approach using CLIP's multimodal capabilities directly.
    More efficient and leverages CLIP's pre-trained vision-language alignment.
    """
    def __init__(self, num_engagement_metrics=NUM_ENGAGEMENT_METRICS, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # Load pre-trained CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP parameters (optional - can unfreeze for full fine-tuning)
        for param in self.clip.parameters():
            param.requires_grad = False
        
        clip_dim = 512  # CLIP embedding dimension
        
        # Regression head
        self.predictor = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_engagement_metrics)
        )
        
    def forward(self, images, input_ids, attention_mask):
        # Get CLIP embeddings
        outputs = self.clip(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use multimodal embeddings (average of image and text)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        fused_embeds = (image_embeds + text_embeds) / 2
        
        # Predict engagement
        predictions = self.predictor(fused_embeds)
        
        return predictions



class MultimodalEngagementPredictor(nn.Module):
    """
    Multimodal model for predicting engagement metrics from images and text.
    Combines CLIP vision encoder with BERT text encoder.
    """
    def __init__(self, num_engagement_metrics=NUM_ENGAGEMENT_METRICS, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # Vision encoder (using CLIP's vision model)
        self.vision_encoder = models.resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Identity()  # Remove final layer
        vision_dim = 2048
        
        # Text encoder (using BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        text_dim = 768
        
        # Projection layers to common dimension
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion and prediction layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for each engagement metric
        self.engagement_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_engagement_metrics)
        ])
        
    def forward(self, images, input_ids, attention_mask):
        # Extract visual features
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_projection(vision_features)
        
        # Extract text features
        text_outputs = self.text_encoder(input_ids=input_ids, 
                                         attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_features = self.text_projection(text_features)
        
        # Fuse features
        fused_features = torch.cat([vision_features, text_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Predict each engagement metric
        predictions = torch.cat([head(fused_features) for head in self.engagement_heads], dim=1)
        
        return predictions
    
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    print(f"inside train epoch... dataloader length is {len(dataloader)}")  # DEBUGGING LINE
    with alive_bar(len(dataloader), title="Training Progress") as bar:
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['engagement'].to(device)
            
            optimizer.zero_grad()
            
            predictions = model(images, input_ids, attention_mask)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            bar()  # Update the progress bar

    # for batch in dataloader:
    #     images = batch['image'].to(device)
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     targets = batch['engagement'].to(device)
        
    #     optimizer.zero_grad()
        
    #     predictions = model(images, input_ids, attention_mask)
    #     loss = criterion(predictions, targets)
        
    #     loss.backward()
    #     optimizer.step()
        
    #     total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['engagement'].to(device)
            
            predictions = model(images, input_ids, attention_mask)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    return total_loss / len(dataloader), all_predictions, all_targets
