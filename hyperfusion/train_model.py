import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
import pickle
import os


FEATURES_DIR = './features'
OUTPUT_DIR = './models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_features():
    """Load preprocessed features"""
    print("Loading features...")
    feature_df = pd.read_parquet(f'{FEATURES_DIR}/feature_data.parquet')
    
    with open(f'{FEATURES_DIR}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Loaded {len(feature_df)} samples")
    print(f"Features shape: {feature_df.shape}")
    print(f"Labels: {label_encoder.classes_}")
    
    return feature_df, label_encoder

# ============================================
# Prepare Feats for Training
# ============================================

def prepare_training_data(feature_df):
    """Prepare features for CatBoost classifier"""
    
    train_data = feature_df[feature_df['train_type'] == 0].reset_index(drop=True)
    val_data = feature_df[feature_df['train_type'] == 1].reset_index(drop=True)
    test_data = feature_df[feature_df['train_type'] == 2].reset_index(drop=True)
    
    # print(f"\nData splits:")
    # print(f"  Train size: {len(train_data)}")
    # print(f"  Val size: {len(val_data)}")
    # print(f"  Test size: {len(test_data)}")
    
    exclude_cols = ['idx', 'label', 'train_type']
    
    # Categorical features
    categorical_features = ['tag_encoded']
    
    
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    
    # print(f"\nUsing {len(feature_cols)} features")
    # print(f"Categorical features: {categorical_features}")
    
    # Extract features and labels
    X_train = train_data[feature_cols]
    y_train = train_data['label']
    
    X_val = val_data[feature_cols]
    y_val = val_data['label']
    
    X_test = test_data[feature_cols]
    y_test = test_data['label']
    
    # Print class distribution
    print("\nClass distribution:")
    print("Train:", y_train.value_counts().sort_index().to_dict())
    print("Val:", y_val.value_counts().sort_index().to_dict())
    print("Test:", y_test.value_counts().sort_index().to_dict())
    
    return X_train, y_train, X_val, y_val, X_test, y_test, categorical_features

# ============================================
# Train CatBoost Classifier
# ============================================

def train_catboost_classifier(X_train, y_train, X_val, y_val, categorical_features):    
    # CatBoost parameters
    params = {
        'iterations': 5,  # TODO: modify
        'learning_rate': 0.03,
        'depth': 6,
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'early_stopping_rounds': 200,
        'task_type': 'GPU',  # Change to 'CPU' if no GPU available
        'devices': '0',
        'verbose': 100,
        'random_seed': 42,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'use_best_model': True
    }
    
    # Create pools
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    
    # Train model
    print("\n" + "="*70)
    print("Training CatBoost Classifier...")
    print("="*70)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, plot=False)
    
    return model

# ============================================
# Evaluate
# ============================================

def evaluate_model(model, X, y, label_encoder, dataset_name="Test"):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"{dataset_name} Set Results:")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y, y_pred, 
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    print("\nPer-class Accuracy:")
    for i, label in enumerate(label_encoder.classes_):
        mask = y == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"  {label}: {class_acc:.4f} ({mask.sum()} samples)")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# ============================================
# K-Fold CV
# ============================================

def cross_validate_model(X, y, categorical_features, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': []
    }
    
    fold = 0
    for train_idx, val_idx in skf.split(X, y):
        print(f"\n{'='*70}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*70}")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = train_catboost_classifier(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            categorical_features
        )
        
        y_pred = model.predict(X_val_fold)
        
        acc = accuracy_score(y_val_fold, y_pred)
        f1_m = f1_score(y_val_fold, y_pred, average='macro')
        f1_w = f1_score(y_val_fold, y_pred, average='weighted')
        
        cv_results['accuracy'].append(acc)
        cv_results['f1_macro'].append(f1_m)
        cv_results['f1_weighted'].append(f1_w)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (Macro): {f1_m:.4f}")
        print(f"  F1 (Weighted): {f1_w:.4f}")
        
        model.save_model(f'{OUTPUT_DIR}/model_fold_{fold}.cbm')
        print(f"  Model saved: {OUTPUT_DIR}/model_fold_{fold}.cbm")
        
        fold += 1
    
    print(f"\n{'='*70}")
    print("Cross Validation Summary:")
    print(f"{'='*70}")
    print(f"Accuracy: {np.mean(cv_results['accuracy']):.4f} ± {np.std(cv_results['accuracy']):.4f}")
    print(f"F1 (Macro): {np.mean(cv_results['f1_macro']):.4f} ± {np.std(cv_results['f1_macro']):.4f}")
    print(f"F1 (Weighted): {np.mean(cv_results['f1_weighted']):.4f} ± {np.std(cv_results['f1_weighted']):.4f}")
    
    return cv_results

# ============================================
# Feature Importance 
# ============================================

# def analyze_feature_importance(model, X, top_n=20):
#     """Analyze and plot feature importance"""
    
#     # Get feature importance
#     feature_importance = model.get_feature_importance()
#     feature_names = X.columns
    
#     # Create dataframe
#     importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': feature_importance
#     }).sort_values('importance', ascending=False)
    
#     print(f"\n{'='*70}")
#     print(f"Top {top_n} Most Important Features:")
#     print(f"{'='*70}")
#     for idx, row in importance_df.head(top_n).iterrows():
#         print(f"  {row['feature']}: {row['importance']:.4f}")
    
#     # Save to file
#     importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
#     print(f"\nFull feature importance saved to: {OUTPUT_DIR}/feature_importance.csv")
    
#     return importance_df

# ============================================
# Main Training Pipeline
# ============================================

def main():
    feature_df, label_encoder = load_features()
    
    print("\nPreparing ...")
    X_train, y_train, X_val, y_val, X_test, y_test, categorical_features = \
        prepare_training_data(feature_df)
    
    print("\n" + "="*70)
    print("Training Model")
    print("="*70)
    model = train_catboost_classifier(X_train, y_train, X_val, y_val, categorical_features)
    
    val_results = evaluate_model(model, X_val, y_val, label_encoder, "Validation")
    
    test_results = evaluate_model(model, X_test, y_test, label_encoder, "Test")
    
    model.save_model(f'{OUTPUT_DIR}/final_model.cbm')
    print(f"\n✓ Final model saved to {OUTPUT_DIR}/final_model.cbm")
    
    # Analyze feature importance
    # importance_df = analyze_feature_importance(model, X_train, top_n=20)
    
    # Cross validation on train data
    print("\n" + "="*70)
    print("Performing 5-Fold CV")
    print("="*70)
    X_train_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    
    cv_results = cross_validate_model(X_train_full, y_train_full, categorical_features, n_splits=5)
    
    results_summary = {
        'validation': val_results,
        'test': test_results,
        'cv': cv_results
    }
    
    with open(f'{OUTPUT_DIR}/results_summary.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    print("\n" + "="*70)
    print("Metrics ::")
    print("="*70)
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1 (Macro): {test_results['f1_macro']:.4f}")
    print(f"Test F1 (Weighted): {test_results['f1_weighted']:.4f}")
    print(f"CV Accuracy: {np.mean(cv_results['accuracy']):.4f} ± {np.std(cv_results['accuracy']):.4f}")
    
    return model, results_summary

if __name__ == "__main__":
    model, results = main()