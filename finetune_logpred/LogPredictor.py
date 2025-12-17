#  Imports
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoProcessor, AutoModel, DebertaV2Tokenizer,
    get_cosine_schedule_with_warmup
)

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All imports successful!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
# Configuration
CONFIG = {
    # Data paths
    'train_json': 'datasets/train_with_eng_fixed.json',
    'val_json': 'datasets/validation_with_eng_fixed.json',
    'test_json': 'datasets/test_with_eng_fixed.json',
    
    'vision_encoder': 'google/siglip-so400m-patch14-384',  
    'text_encoder': 'microsoft/deberta-v3-large',
    
    'hidden_dim': 768, 
    'num_attention_heads': 12,  
    'num_fusion_layers': 2,  
    'num_early_fusion_layers': 1, 
    'dropout': 0.3,
    
    'num_targets': 1,
    'target_name': 'log_virality',
    
    'virality_weights': None,
    'log_epsilon': 1.0,
    
    
    'batch_size': 16, 
    'gradient_accumulation_steps': 3,
    'num_epochs': 50,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'warmup_ratio': 0.1,
    'use_fp16': True,
    
    'max_title_length': 64,
    'max_context_length': 256,
    
    # hybrid of huber(0.3) & MSE LOSS (0.7)
    'loss_type': 'hybrid',
    'hybrid_alpha': 0.7, 
    'huber_delta': 1.0,
    
    'freeze_vision_encoder': False,
    'freeze_text_encoder': False,  
    'freeze_epochs': 0,  
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 12,
    'pin_memory': True,
    'prefetch_factor': 4,
    'persistent_workers': True,
    'seed': 42,
    
    'checkpoint_dir': '/work/classtmp/rohail03/hardSoTA/checkpoints',
    'results_dir': './results',
}

Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True, parents=True)
Path(CONFIG['results_dir']).mkdir(exist_ok=True, parents=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

set_seed(CONFIG['seed'])

print(f"Vision: {CONFIG['vision_encoder']}")
print(f"Text: {CONFIG['text_encoder']}")
print(f"Device: {CONFIG['device']}")
print(f"Target: {CONFIG['target_name']}")
print(f"Loss: {CONFIG['loss_type']} (alpha={CONFIG['hybrid_alpha']})")

# Helper function to compute correlation weights
def compute_virality_weights(data_list):
    views = []
    likes = []
    comments = []
    shares = []
    favorites = []
    
    for item in data_list:
        eng = item.get('engagement', {})
        views.append(float(eng.get('views', 0)))
        likes.append(float(eng.get('likes', 0)))
        comments.append(float(eng.get('comments', 0)))
        shares.append(float(eng.get('shares', 0)))
        favorites.append(float(eng.get('favorites', 0)))
    
    views = np.array(views)
    likes = np.array(likes)
    comments = np.array(comments)
    shares = np.array(shares)
    favorites = np.array(favorites)
    
    corr_views_likes = np.corrcoef(views, likes)[0, 1] if len(views) > 1 else 0.5
    corr_views_comments = np.corrcoef(views, comments)[0, 1] if len(views) > 1 else 0.5
    corr_views_shares = np.corrcoef(views, shares)[0, 1] if len(views) > 1 else 0.5
    corr_views_favorites = np.corrcoef(views, favorites)[0, 1] if len(views) > 1 else 0.5
    
    # Handle NaN 
    corr_views_likes = 0.5 if np.isnan(corr_views_likes) else corr_views_likes
    corr_views_comments = 0.5 if np.isnan(corr_views_comments) else corr_views_comments
    corr_views_shares = 0.5 if np.isnan(corr_views_shares) else corr_views_shares
    corr_views_favorites = 0.5 if np.isnan(corr_views_favorites) else corr_views_favorites
    
    weights = {
        'views': 1.0,
        'likes': max(0, 1 - corr_views_likes),
        'comments': max(0, 1 - corr_views_comments),
        'shares': max(0, 1 - corr_views_shares),
        'favorites': max(0, 1 - corr_views_favorites),
    }
    
    return weights

def calculate_virality(engagement, weights, log_epsilon=1.0):
    virality = (
        weights['views'] * engagement.get('views', 0) +
        weights['likes'] * engagement.get('likes', 0) +
        weights['comments'] * engagement.get('comments', 0) +
        weights['shares'] * engagement.get('shares', 0) +
        weights['favorites'] * engagement.get('favorites', 0)
    )
    return np.log(virality + log_epsilon)

class ViralityRegressionDataset(Dataset):
    def __init__(self, data_list, vision_processor, text_tokenizer, config, weights=None):
        self.data = data_list
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.config = config
        
        if weights is None:
            self.weights = compute_virality_weights(data_list)
            if config['virality_weights'] is None:
                config['virality_weights'] = self.weights
        else:
            self.weights = weights
        
        self._compute_statistics()
    
    def _compute_statistics(self):
        virality_scores = []
        
        for item in self.data:
            eng = item.get('engagement', {})
            log_virality = calculate_virality(eng, self.weights, self.config['log_epsilon'])
            virality_scores.append(log_virality)
        
        virality_scores = np.array(virality_scores)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            image = Image.open(item['image']).convert('RGB')
            image_inputs = self.vision_processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].squeeze(0)
        except:
            blank = Image.new('RGB', (384, 384), color='gray')
            image_inputs = self.vision_processor(images=blank, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        title = item.get('title', '')
        title_encoding = self.text_tokenizer(
            title,
            max_length=self.config['max_title_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        tag = item.get('tag', '')
        description = item.get('description', '')
        context = f"{tag} [SEP] {description}"
        
        context_encoding = self.text_tokenizer(
            context,
            max_length=self.config['max_context_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        eng = item.get('engagement', {})
        log_virality = calculate_virality(eng, self.weights, self.config['log_epsilon'])
        target = torch.tensor([log_virality], dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,
            'title_input_ids': title_encoding['input_ids'].squeeze(0),
            'title_attention_mask': title_encoding['attention_mask'].squeeze(0),
            'context_input_ids': context_encoding['input_ids'].squeeze(0),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(0),
            'target': target
        }

print("✓ Dataset class defined!")

class HybridViralityLoss(nn.Module):
    def __init__(self, alpha=0.7, delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        huber_loss = self.huber(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * huber_loss

print("✓ Hybrid loss function defined!")


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query, key_value, kv_mask=None):
        attn_output, attn_weights = self.multihead_attn(
            query, key_value, key_value,
            key_padding_mask=kv_mask,
            need_weights=True
        )
        query = self.norm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        return query, attn_weights


class EarlyCrossModalFusion(nn.Module):
    # !! Bidirectional fusion
    def __init__(self, hidden_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.cross_attn_v2t = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.cross_attn_t2v = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_feats, text_feats, text_mask=None):
        v_fused, _ = self.cross_attn_v2t(vision_feats, text_feats, text_mask)
        v_fused = self.vision_norm(v_fused)
        
        t_fused, _ = self.cross_attn_t2v(text_feats, vision_feats, None)
        t_fused = self.text_norm(t_fused)
        
        return v_fused, t_fused


class TransformerFusionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities=3, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, *modality_features):
        concat = torch.cat(modality_features, dim=-1)
        gates = self.gate(concat)
        fused = sum(g.unsqueeze(-1) * f for g, f in zip(gates.unbind(dim=-1), modality_features))
        return fused, gates

print("✓ Fusion modules defined!")


class EnhancedViralityRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print(f"Loading vision: {config['vision_encoder']}...")
        full_model = AutoModel.from_pretrained(config['vision_encoder'])
        # Extract only the vision encoder from SigLIP! (SigLIP takes image and text data at once)
        if hasattr(full_model, 'vision_model'):
            self.vision_encoder = full_model.vision_model
            vision_dim = full_model.config.vision_config.hidden_size
        else:
            print("Couldn't load vision model!")
            exit(1)
        
        print(f"Loading text: {config['text_encoder']}...")
        self.title_encoder = AutoModel.from_pretrained(config['text_encoder'])
        self.context_encoder = AutoModel.from_pretrained(config['text_encoder'])
        text_dim = self.title_encoder.config.hidden_size
        
        if config['freeze_vision_encoder']:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        if config['freeze_text_encoder']:
            for p in self.title_encoder.parameters():
                p.requires_grad = False
            for p in self.context_encoder.parameters():
                p.requires_grad = False
        
        hidden_dim = config['hidden_dim']
        dropout = config['dropout']
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.title_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.context_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        num_heads = config['num_attention_heads']
        self.early_fusion_layers = nn.ModuleList([
            EarlyCrossModalFusion(hidden_dim, num_heads, dropout)
            for _ in range(config['num_early_fusion_layers'])
        ])
        
        self.vision_to_text = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.text_to_vision = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.title_context_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        self.fusion_layers = nn.ModuleList([
            TransformerFusionLayer(hidden_dim, num_heads, dropout)
            for _ in range(config['num_fusion_layers'])
        ])
        
        self.gated_fusion = GatedFusion(hidden_dim, 3, dropout)
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.vision_proj, self.title_proj, self.context_proj, self.regressor]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def unfreeze_encoders(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = True
        for p in self.title_encoder.parameters():
            p.requires_grad = True
        for p in self.context_encoder.parameters():
            p.requires_grad = True
    
    def forward(self, pixel_values, title_input_ids, title_attention_mask,
                context_input_ids, context_attention_mask):
        
        vision_out = self.vision_encoder(pixel_values=pixel_values)
        vision_feats = self.vision_proj(vision_out.last_hidden_state)
        
        title_out = self.title_encoder(input_ids=title_input_ids, attention_mask=title_attention_mask)
        title_feats = self.title_proj(title_out.last_hidden_state)
        
        context_out = self.context_encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)
        context_feats = self.context_proj(context_out.last_hidden_state)
        
        text_combined = torch.cat([title_feats, context_feats], dim=1)
        text_mask_combined = torch.cat([title_attention_mask == 0, context_attention_mask == 0], dim=1)
        
        v_fused = vision_feats
        t_fused = text_combined
        
        for early_fusion_layer in self.early_fusion_layers:
            v_fused, t_fused = early_fusion_layer(v_fused, t_fused, text_mask_combined)
        
        title_len = title_feats.size(1)
        title_fused = t_fused[:, :title_len, :]
        context_fused = t_fused[:, title_len:, :]
        
        vision_attn, _ = self.vision_to_text(v_fused, t_fused, text_mask_combined)
        title_attn, _ = self.text_to_vision(title_fused, v_fused)
        context_attn, _ = self.text_to_vision(context_fused, v_fused)
        title_ctx_fused, _ = self.title_context_attn(title_attn, context_attn)
        
        vision_pooled = vision_attn.mean(dim=1)
        
        title_mask_exp = title_attention_mask.unsqueeze(-1).expand(title_ctx_fused.size())
        title_pooled = (title_ctx_fused * title_mask_exp).sum(1) / title_mask_exp.sum(1).clamp(min=1)
        
        context_mask_exp = context_attention_mask.unsqueeze(-1).expand(context_attn.size())
        context_pooled = (context_attn * context_mask_exp).sum(1) / context_mask_exp.sum(1).clamp(min=1)
        
        fused, _ = self.gated_fusion(vision_pooled, title_pooled, context_pooled)
        
        fused_seq = fused.unsqueeze(1)
        for layer in self.fusion_layers:
            fused_seq = layer(fused_seq)
        fused_final = fused_seq.squeeze(1)
        
        prediction = self.regressor(fused_final)
        return prediction

print("✓ Model arch defined!")

print("Loading data...")

with open(CONFIG['train_json'], 'r') as f:
    train_data = json.load(f)

with open(CONFIG['val_json'], 'r') as f:
    val_data = json.load(f)

with open(CONFIG['test_json'], 'r') as f:
    test_data = json.load(f)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

print("Initializing processors...")

vision_processor = AutoProcessor.from_pretrained(CONFIG['vision_encoder'])
text_tokenizer = DebertaV2Tokenizer.from_pretrained(CONFIG['text_encoder'])

print("✓ Processors loaded!")

print("Creating datasets...")
train_dataset = ViralityRegressionDataset(train_data, vision_processor, text_tokenizer, CONFIG)
val_dataset = ViralityRegressionDataset(val_data, vision_processor, text_tokenizer, CONFIG, weights=CONFIG['virality_weights'])
test_dataset = ViralityRegressionDataset(test_data, vision_processor, text_tokenizer, CONFIG,weights=CONFIG['virality_weights'])

print("✓ Datasets created!")

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=CONFIG['pin_memory'],
    prefetch_factor=CONFIG['prefetch_factor'],
    persistent_workers=CONFIG['persistent_workers']
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=CONFIG['pin_memory'],
    prefetch_factor=CONFIG['prefetch_factor'],
    persistent_workers=CONFIG['persistent_workers']
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=CONFIG['pin_memory'],
    prefetch_factor=CONFIG['prefetch_factor'],
    persistent_workers=CONFIG['persistent_workers']
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

print("Initializing model...")

model = EnhancedViralityRegressor(CONFIG).to(CONFIG['device'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


if CONFIG['loss_type'] == 'hybrid':
    criterion = HybridViralityLoss(
        alpha=CONFIG['hybrid_alpha'],
        delta=CONFIG['huber_delta']
    )
elif CONFIG['loss_type'] == 'mse':
    criterion = nn.MSELoss()
elif CONFIG['loss_type'] == 'mae':
    criterion = nn.L1Loss()
elif CONFIG['loss_type'] == 'huber':
    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'])
else:
    criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation_steps']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

if CONFIG['use_fp16']:
    scaler = torch.cuda.amp.GradScaler()

print("✓ Training setup complete!")

def train_epoch(model, loader, optimizer, criterion, scheduler, scaler, config):
    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(config['device'])
        title_ids = batch['title_input_ids'].to(config['device'])
        title_mask = batch['title_attention_mask'].to(config['device'])
        context_ids = batch['context_input_ids'].to(config['device'])
        context_mask = batch['context_attention_mask'].to(config['device'])
        target = batch['target'].to(config['device'])
        
        if config['use_fp16']:
            with torch.cuda.amp.autocast():
                pred = model(pixel_values, title_ids, title_mask, context_ids, context_mask)
                loss = criterion(pred, target) / config['gradient_accumulation_steps']
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            pred = model(pixel_values, title_ids, title_mask, context_ids, context_mask)
            loss = criterion(pred, target) / config['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * config['gradient_accumulation_steps']
        
        pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(config['device'])
            title_ids = batch['title_input_ids'].to(config['device'])
            title_mask = batch['title_attention_mask'].to(config['device'])
            context_ids = batch['context_input_ids'].to(config['device'])
            context_mask = batch['context_attention_mask'].to(config['device'])
            target = batch['target'].to(config['device'])
            
            if config['use_fp16']:
                with torch.cuda.amp.autocast():
                    pred = model(pixel_values, title_ids, title_mask, context_ids, context_mask)
                    loss = criterion(pred, target)
            else:
                pred = model(pixel_values, title_ids, title_mask, context_ids, context_mask)
                loss = criterion(pred, target)
            
            total_loss += loss.item()
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.vstack(all_preds).flatten()
    all_targets = np.vstack(all_targets).flatten()
    
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mse)
    
    mape = np.mean(np.abs((all_targets - all_preds) / (np.abs(all_targets) + 1e-8))) * 100
    
    avg_loss = total_loss / len(loader)
    
    metrics = {
        'loss': avg_loss,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    return metrics, all_preds, all_targets

print("✓ Training functions defined!")

# Training Loop
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

history = {
    'train_loss': [],
    'val_loss': [],
    'val_r2': [],
    'val_mae': []
}

best_val_r2 = -float('inf')

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    print("-"*80)
    
    if epoch == CONFIG['freeze_epochs'] and CONFIG['freeze_epochs'] > 0:
        model.unfreeze_encoders()
    
    # Train
    train_loss = train_epoch(
        model, train_loader, optimizer, criterion, scheduler,
        scaler if CONFIG['use_fp16'] else None, CONFIG
    )
    
    # Validate
    val_metrics, _, _ = evaluate(model, val_loader, criterion, CONFIG)
    
    # Update history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_metrics['loss'])
    history['val_r2'].append(val_metrics['r2'])
    history['val_mae'].append(val_metrics['mae'])
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Val R²: {val_metrics['r2']:.4f}")
    print(f"Val MAE: {val_metrics['mae']:.4f}")
    print(f"Val RMSE: {val_metrics['rmse']:.4f}")
    print(f"Val MAPE: {val_metrics['mape']:.2f}%")
    
    if val_metrics['r2'] > best_val_r2:
        best_val_r2 = val_metrics['r2']
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': CONFIG
        }
        torch.save(checkpoint, Path(CONFIG['checkpoint_dir']) / 'best_model_enhanced.pth')
        print(f"✓ Saved best model (Val R²: {val_metrics['r2']:.4f})")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Best Val R²: {best_val_r2:.4f}")

# Plot Training Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss (Hybrid)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['val_r2'], label='Val R²', marker='o', color='green')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Validation R² Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(history['val_mae'], label='Val MAE', marker='o', color='orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MAE')
axes[2].set_title('Validation MAE')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(CONFIG['results_dir']) / 'training_curves_enhanced.png', dpi=300)
plt.show()

print("\n✓ Training curves saved!")

print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80 + "\n")

# Load best model
checkpoint = torch.load(Path(CONFIG['checkpoint_dir']) / 'best_model_enhanced.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")

test_metrics, test_preds, test_targets = evaluate(model, test_loader, criterion, CONFIG)

print("\nTEST SET RESULTS:")
print("-"*80)
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test R²: {test_metrics['r2']:.4f}")
print(f"Test MSE: {test_metrics['mse']:.4f}")
print(f"Test MAE: {test_metrics['mae']:.4f}")
print(f"Test RMSE: {test_metrics['rmse']:.4f}")
print(f"Test MAPE: {test_metrics['mape']:.2f}%")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(test_targets, test_preds, alpha=0.5, s=10)
axes[0].plot([test_targets.min(), test_targets.max()], 
             [test_targets.min(), test_targets.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Log-Virality')
axes[0].set_ylabel('Predicted Log-Virality')
axes[0].set_title(f'Test Set Predictions (R²={test_metrics["r2"]:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

residuals = test_targets - test_preds
axes[1].scatter(test_preds, residuals, alpha=0.5, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Log-Virality')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(CONFIG['results_dir']) / 'test_predictions_enhanced.png', dpi=300)
plt.show()
