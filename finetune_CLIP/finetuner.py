import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import json
import pickle

from utils import *


def train_engagement_model(
    train_path,
    val_path,
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Complete training pipeline for engagement prediction
    
    Args:
        data_path: Path to JSON file with data
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
    """
    # Load data
    print(device)
    with open(train_path, 'r') as f:
        train_data = json.load(f)
        
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    # Initialize normalizer
    normalizer = EngagementNormalizer()
    normalizer.fit([item['engagement'] for item in train_data])
    
    # Data transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    train_dataset = PixelRecDataset(train_data, image_transform, tokenizer, normalizer)
    val_dataset = PixelRecDataset(val_data, image_transform, tokenizer, normalizer)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    
    # Initialize model
    # model = MultimodalEngagementPredictor().to(device)
    
    # Using CLIP-based model
    model = CLIPBasedEngagementPredictor(normalizer = normalizer).to(device) #TODO: model changes made here
    
    #Using CLIP Vision + BERT Text model
    # model = CLIPVisionBERTTextPredictor(normalizer = normalizer).to(device)
    
    # Loss function (MSE for regression)
    class ToleranceMSE(nn.Module):
        def __init__(self, tolerance=0.01, reduction='mean', eps=1e-8):
            super().__init__()
            self.tolerance = tolerance
            self.reduction = reduction
            self.eps = eps
            self.last_accuracy = 0.0

        def forward(self, preds, targets):
            rel_err = torch.abs(preds - targets) / (torch.abs(targets) + self.eps)
            with torch.no_grad():
                correct = (rel_err <= self.tolerance).float()
                self.last_accuracy = correct.mean().item()
            mask = (rel_err > self.tolerance).float()
            se = (preds - targets) ** 2
            masked_se = se * mask
            if self.reduction == 'mean':
                return masked_se.mean()
            elif self.reduction == 'sum':
                return masked_se.sum()
            else:
                return masked_se

    criterion = ToleranceMSE(tolerance=0.01, reduction='mean') #TODO: Modify tolerance param
    
    # criterion = nn.MSELoss()  # Original Loss function
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                           weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, 
                                criterion, device)
        
        # Validate
        val_loss, predictions, targets = evaluate(model, val_loader, 
                                                  criterion, device)
        
        # Calculate metrics
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {mae:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_engagement_model.pth')
            
            # Save normalizer
            with open('engagement_normalizer.pkl', 'wb') as f:
                pickle.dump(normalizer, f)
            
            print(f"Model saved with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return model, normalizer


# Inference function
def predict_engagement(model, normalizer, image_path, title, tag, description, device='cuda'):
    """
    Predict engagement metrics for a new post
    
    Returns:
        dict: Predicted engagement metrics (likes, comments, shares, views, favorites)
    """
    from PIL import Image
    from torchvision import transforms
    from transformers import BertTokenizer
    
    model.eval()
    
    # Prepare image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0).to(device)
    
    # Prepare text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = f"Title: {title} Tag: {tag} Description: {description}"
    encoding = tokenizer(text, padding='max_length', truncation=True, 
                        max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(image, input_ids, attention_mask)
    
    # Denormalize
    predictions_np = predictions.cpu().numpy()
    denormalized = normalizer.inverse_transform(predictions_np)[0]
    
    # Format output
    engagement = {
        'likes': int(max(0, denormalized[0])),
        'comments': int(max(0, denormalized[1])),
        'shares': int(max(0, denormalized[2])),
        'views': int(max(0, denormalized[3])),
        'favorites': int(max(0, denormalized[4]))
    }
    
    return engagement

if __name__ == "__main__":
    model, normalizer = train_engagement_model(
        train_path = '../data/train.json', #TODO: modify paths and params accordingly
        val_path = '../data/test.json',
        batch_size=32, 
        num_epochs=50, 
        learning_rate=1e-4
    )
