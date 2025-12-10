import pandas as pd
import numpy as np
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import os

def load_data(train_path, val_path, test_path):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    # Add identifier
    train_df['train_type'] = 0
    val_df['train_type'] = 1
    test_df['train_type'] = 2
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df):
    all_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    
    label_encoder = LabelEncoder()
    all_df['label_encoded'] = label_encoder.fit_transform(all_df['engagement_label'])
    
    return all_df, label_encoder

# ============================================
# Feature Extraction with CLIP
# ============================================

class EngagementDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        try:
            img = Image.open(row['image']).convert("RGB")
        except:
            # Create a blank image if the image file is not found
            print(f"Warning: Image at {row['image']} not found. Using blank image).")
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Combine text fields
        text = f"{row['title']} {row['tag']} {row['description']}"
        
        return {
            'image': img,
            'text': text,
            'label': row['label_encoded']
        }

def extract_clip_features(df, model, processor, batch_size=32, device='cuda'):
    dataset = EngagementDataset(df, processor)
    
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        return {
            'pixel_values': inputs['pixel_values'],
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': torch.tensor(labels)
        }
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 because windows stinks at multiprocessing
        collate_fn=collate_fn
    )
    
    image_features_list = []
    text_features_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting CLIP features"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get features
            image_features = model.get_image_features(pixel_values=pixel_values)
            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            image_features_list.append(image_features.cpu().numpy())
            text_features_list.append(text_features.cpu().numpy())
    
    image_features = np.vstack(image_features_list)
    text_features = np.vstack(text_features_list)
    
    return image_features, text_features

# ============================================
# Create Feature DataFrame
# ============================================

def create_feature_dataframe(df, image_features, text_features):
    engagement_data = df['engagement'].apply(pd.Series)
    
    features_dict = {
        'idx': df.index,
        'label': df['label_encoded'],
        'train_type': df['train_type'],
        
        'likes': engagement_data['likes'],
        'comments': engagement_data['comments'],
        'shares': engagement_data['shares'],
        'views': engagement_data['views'],
        'favorites': engagement_data['favorites'],
    }
    
    # Create DataFrame from base features
    feature_df = pd.DataFrame(features_dict)
    
    # Add log-transformed engagement metrics
    log_features = pd.DataFrame({
        'log_likes': np.log1p(feature_df['likes']),
        'log_comments': np.log1p(feature_df['comments']),
        'log_shares': np.log1p(feature_df['shares']),
        'log_views': np.log1p(feature_df['views']),
        'log_favorites': np.log1p(feature_df['favorites'])
    })
    
    # Add derived features
    derived_features = pd.DataFrame({
        'engagement_rate': (
            (feature_df['likes'] + feature_df['comments'] + feature_df['shares']) / 
            (feature_df['views'] + 1) # +1 to avoid division by zero
        ),
        'comment_like_ratio': feature_df['comments'] / (feature_df['likes'] + 1),
        'favorite_like_ratio': feature_df['favorites'] / (feature_df['likes'] + 1)
    })
    
    # Text features
    text_stat_features = pd.DataFrame({
        'title_len': df['title'].str.len(),
        'title_words': df['title'].str.split().str.len(),
        'desc_len': df['description'].str.len(),
        'desc_words': df['description'].str.split().str.len()
    })
    
    # Encode categorical tag
    tag_encoder = LabelEncoder()
    tag_features = pd.DataFrame({
        'tag_encoded': tag_encoder.fit_transform(df['tag'])
    })
    
    # Create CLIP feature DataFrames
    img_feat_cols = {f'img_feat_{i}': image_features[:, i] for i in range(image_features.shape[1])}
    img_feat_df = pd.DataFrame(img_feat_cols)
    
    text_feat_cols = {f'text_feat_{i}': text_features[:, i] for i in range(text_features.shape[1])}
    text_feat_df = pd.DataFrame(text_feat_cols)
    
    # Calculate image-text similarity
    similarity = np.sum(image_features * text_features, axis=1) / (
        np.linalg.norm(image_features, axis=1) * np.linalg.norm(text_features, axis=1)
    )
    similarity_df = pd.DataFrame({'img_text_similarity': similarity})
    
    # Concatenate all features 
    feature_df = pd.concat([
        feature_df,
        log_features,
        derived_features,
        text_stat_features,
        tag_features,
        img_feat_df,
        text_feat_df,
        similarity_df
    ], axis=1)
    
    return feature_df


def main():
    # TODO: update these paths!!
    TRAIN_PATH = '../../data/binned/baby_test.json'
    VAL_PATH = '../../data/binned/baby_test.json'
    TEST_PATH = '../../data/binned/baby_test.json'
    OUTPUT_DIR = './features'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    
    train_df, val_df, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    all_data, label_encoder = prepare_data(train_df, val_df, test_df)
    # print(f"\nTotal data shape: {all_data.shape}")
    # print(f"\nLabel distribution:")
    # print(all_data.groupby('train_type')['engagement_label'].value_counts().unstack(fill_value=0))
    
    # Save label encoder
    import pickle
    with open(f'{OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Load CLIP model
    print("\nLoading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Extract features
    print("\nExtracting CLIP features...")
    image_features, text_features = extract_clip_features(
        all_data, model, processor, batch_size=32, device=device
    )
    
    # Save raw features
    np.save(f'{OUTPUT_DIR}/image_features.npy', image_features)
    np.save(f'{OUTPUT_DIR}/text_features.npy', text_features)
    # print(f"Image features shape: {image_features.shape}")
    # print(f"Text features shape: {text_features.shape}")
    
    print("\nCreating feature dataframe...")
    feature_df = create_feature_dataframe(all_data, image_features, text_features)
    
    # Save features
    feature_df.to_parquet(f'{OUTPUT_DIR}/feature_data.parquet', index=False)
    feature_df.to_csv(f'{OUTPUT_DIR}/feature_data.csv', index=False)
    
    # print(f"\nFeature dataframe shape: {feature_df.shape}")
    # print(f"\nFeatures saved to {OUTPUT_DIR}/")
    # print("\nSample features:")
    # print(feature_df.head())
    
    # # Print the feature statistics by split and class
    # print("\n" + "="*50)
    # print("Feature Statistics by Split and Engagement Label:")
    # print("="*50)
    # split_names = {0: 'Train', 1: 'Validation', 2: 'Test'}
    # for split_type in [0, 1, 2]:
    #     split_data = all_data[all_data['train_type'] == split_type]
    #     print(f"\n{split_names[split_type]} Set:")
    #     for label in split_data['engagement_label'].unique():
    #         mask = (all_data['train_type'] == split_type) & (all_data['engagement_label'] == label)
    #         print(f"  {label}:")
    #         print(f"    Count: {mask.sum()}")
    #         print(f"    Avg views: {all_data[mask]['engagement'].apply(lambda x: x['views']).mean():.0f}")
    #         print(f"    Avg likes: {all_data[mask]['engagement'].apply(lambda x: x['likes']).mean():.0f}")
    
    return feature_df, label_encoder

if __name__ == "__main__":
    feature_df, label_encoder = main()