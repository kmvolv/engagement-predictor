import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import json
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel
from sklearn.preprocessing import LabelEncoder


def load_model_and_encoders():
    print("Loading model...")
    model = CatBoostClassifier()
    model.load_model('./models/final_model.cbm')
    
    with open('./features/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded successfully")
    print(f"Predicting {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
    
    return model, label_encoder


def extract_clip_features_single(image_path, title, tag, description, 
                                 clip_model, clip_processor, device='cuda'):
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # Combine text
    text = f"{title} {tag} {description}"
    
    # Process with CLIP
    inputs = clip_processor(
        text=[text],
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract features
    clip_model.eval()
    with torch.no_grad():
        image_features = clip_model.get_image_features(
            pixel_values=inputs['pixel_values']
        ).cpu().numpy().flatten()
        
        text_features = clip_model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        ).cpu().numpy().flatten()
    
    # Calculate similarity bw image and text
    similarity = np.dot(image_features, text_features) / (
        np.linalg.norm(image_features) * np.linalg.norm(text_features) + 1e-8
    )
    
    return image_features, text_features, similarity

def prepare_features(sample, clip_model, clip_processor, 
                    tag_encoder=None, device='cuda'):
    
    image_features, text_features, similarity = extract_clip_features_single(
        sample['image'],
        sample['title'],
        sample['tag'],
        sample['description'],
        clip_model,
        clip_processor,
        device
    )
    
    features = {}
    
    # Add CLIP features
    for i, val in enumerate(image_features):
        features[f'img_feat_{i}'] = val
    
    for i, val in enumerate(text_features):
        features[f'text_feat_{i}'] = val
    
    features['img_text_similarity'] = similarity
    
    features['title_len'] = len(sample['title'])
    features['title_words'] = len(sample['title'].split())
    features['desc_len'] = len(sample['description'])
    features['desc_words'] = len(sample['description'].split())
    
    if 'engagement' in sample:
        eng = sample['engagement']
        features['likes'] = eng.get('likes', 0)
        features['comments'] = eng.get('comments', 0)
        features['shares'] = eng.get('shares', 0)
        features['views'] = eng.get('views', 0)
        features['favorites'] = eng.get('favorites', 0)
        
        # Log transforms
        features['log_likes'] = np.log1p(eng.get('likes', 0))
        features['log_comments'] = np.log1p(eng.get('comments', 0))
        features['log_shares'] = np.log1p(eng.get('shares', 0))
        features['log_views'] = np.log1p(eng.get('views', 0))
        features['log_favorites'] = np.log1p(eng.get('favorites', 0))
        
        # Derived features
        features['engagement_rate'] = (
            (eng.get('likes', 0) + eng.get('comments', 0) + eng.get('shares', 0)) / 
            (eng.get('views', 1))
        )
        features['comment_like_ratio'] = eng.get('comments', 0) / (eng.get('likes', 1))
        features['favorite_like_ratio'] = eng.get('favorites', 0) / (eng.get('likes', 1))
    else:
        # placeholders
        for metric in ['likes', 'comments', 'shares', 'views', 'favorites']:
            features[metric] = 0
            features[f'log_{metric}'] = 0
        features['engagement_rate'] = 0
        features['comment_like_ratio'] = 0
        features['favorite_like_ratio'] = 0
    
    features['tag_encoded'] = 0  
    
    return features

# ============================================
# Prediction !
# ============================================

def predict_single(sample, model, label_encoder, clip_model, clip_processor, device='cuda'):
    """Make prediction for a single sample"""
    
    # Extract features
    features = prepare_features(sample, clip_model, clip_processor, device=device)
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # Ensure tag_encoded is integer (categorical feature)
    features_df['tag_encoded'] = features_df['tag_encoded'].astype(int)
    
    # Create Pool with categorical features specified
    from catboost import Pool
    pred_pool = Pool(features_df, cat_features=['tag_encoded'])
    
    
    pred_class = model.predict(pred_pool)[0]
    pred_proba = model.predict_proba(pred_pool)[0]
    
    
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    
    result = {
        'predicted_label': pred_label,
        'predicted_class': int(pred_class),
        'confidence': float(pred_proba[pred_class]),
        'probabilities': {
            label: float(prob) 
            for label, prob in zip(label_encoder.classes_, pred_proba)
        }
    }
    
    # Add true label if available
    if 'engagement_label' in sample:
        result['true_label'] = sample['engagement_label']
        result['correct'] = (pred_label == sample['engagement_label'])
    
    return result

def predict_batch(samples, model, label_encoder, clip_model, clip_processor, device='cuda'):
    """Make predictions for multiple samples"""
    
    print(f"Processing {len(samples)} samples...")
    
    features_list = []
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")
        features = prepare_features(sample, clip_model, clip_processor, device=device)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Ensure tag_encoded is integer (categorical feature)
    features_df['tag_encoded'] = features_df['tag_encoded'].astype(int)
    
    # Create Pool with categorical features specified
    from catboost import Pool
    pred_pool = Pool(features_df, cat_features=['tag_encoded'])
    
    pred_classes = model.predict(pred_pool)
    pred_probas = model.predict_proba(pred_pool)
    
    results = []
    for i, (pred_class, pred_proba) in enumerate(zip(pred_classes, pred_probas)):
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        
        result = {
            'sample_idx': i,
            'predicted_label': pred_label,
            'predicted_class': int(pred_class),
            'confidence': float(pred_proba[pred_class]),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(label_encoder.classes_, pred_proba)
            }
        }
        
        if 'engagement_label' in samples[i]:
            result['true_label'] = samples[i]['engagement_label']
            result['correct'] = (pred_label == samples[i]['engagement_label'])
        
        results.append(result)
    
    return results


def main():
    """Example usage of the inference script"""
    
    print("="*70)
    print("Engagement Prediction - Inference")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load model and CLIP
    model, label_encoder = load_model_and_encoders()
    
    print("\nLoading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    
    print("\n" + "="*70)
    print("Prediction for an image")
    print("="*70)
    
    sample = {
        "image": "D:/pixelrec_imgs/cover/i308171.jpg",
        "title": "Brilliance Yu rare angry anger, soul torture: you know what is a star?",
        "tag": "Celebrities Mix",
        "description": "Organize is not easy, please Sanlian Hua Chenyu rare angry anger, soul torture: you know what is a star?"
    }
    
    result = predict_single(sample, model, label_encoder, clip_model, clip_processor, device)
    
    print(f"\nPredicted Label: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nClass Probabilities:")
    for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()