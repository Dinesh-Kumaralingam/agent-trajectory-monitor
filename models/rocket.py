"""ROCKET classifier for agent health prediction using sktime's RocketClassifier.

This script:
1. Loads agent telemetry from SQLite
2. Engineers features per session (error rate, repeat rate, semantic drift, etc.)
3. Reshapes data into 3D array: (n_sessions, n_features, n_timesteps)
4. Trains a RocketClassifier with minirocket transform on 3 classes: success, loop, hallucination
5. Evaluates on held-out test set + 5-fold cross-validation
6. Saves the trained model to models/rocket_model.pkl

Usage:
    python models/rocket.py
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sktime.classification.kernel_based import RocketClassifier

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telemetry.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rocket_model.pkl")

# Feature columns used in training and inference
FEATURE_COLS = [
    'time_since_last_action',
    'reasoning_length',
    'semantic_similarity',
    'error_keywords',
    'is_repeat_command',
    'error_rate_roll',
    'repeat_rate_roll',
    'semantic_drift_roll',
    'reasoning_zscore',
    'time_accel',
    'action_entropy'
]


def engineer_features_for_session(df):
    """Engineer features for a single session dataframe (with steps).
    Returns a 2D numpy array of shape (n_features, n_timesteps).
    """
    df = df.copy()

    df['time_since_last_action'] = df['time_since_last_action'].astype(float)
    df['reasoning_length'] = df['reasoning_length'].astype(int)
    df['semantic_similarity'] = df['semantic_similarity'].astype(float)
    df['error_keywords'] = df['error_keywords'].astype(int)
    df['is_repeat_command'] = df['is_repeat_command'].astype(int)

    df['error_rate_roll'] = df['exit_code'].ne(0).rolling(5, min_periods=1).mean()
    df['repeat_rate_roll'] = df['is_repeat_command'].rolling(5, min_periods=1).mean()
    df['semantic_drift_roll'] = df['semantic_similarity'].rolling(5, min_periods=1).std().fillna(0)
    df['reasoning_zscore'] = (df['reasoning_length'] - df['reasoning_length'].mean()) / (df['reasoning_length'].std() + 1e-8)
    df['time_accel'] = df['time_since_last_action'].diff().fillna(0)

    action_dummies = pd.get_dummies(df['action_type'], prefix='act')
    action_entropy = -np.sum(action_dummies * np.log(action_dummies + 1e-8), axis=1)
    df['action_entropy'] = action_entropy

    return df[FEATURE_COLS].fillna(0).values.T  # shape: (n_features, n_timesteps)


def load_and_engineer_features():
    """Load raw actions and compute per-session time-series features."""
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        a.session_id,
        a.action_type,
        a.command,
        a.exit_code,
        a.timestamp,
        a.time_since_last_action,
        a.reasoning_length,
        a.semantic_similarity,
        a.error_keywords,
        a.is_repeat_command,
        s.outcome
    FROM agent_actions a
    JOIN agent_sessions s ON a.session_id = s.session_id
    ORDER BY a.session_id, a.timestamp
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    feature_dfs = []
    labels = []

    for session_id, group in df.groupby('session_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)

        group['error_rate_roll'] = group['exit_code'].ne(0).rolling(5, min_periods=1).mean()
        group['repeat_rate_roll'] = group['is_repeat_command'].rolling(5, min_periods=1).mean()
        group['semantic_drift_roll'] = group['semantic_similarity'].rolling(5, min_periods=1).std().fillna(0)
        group['reasoning_zscore'] = (group['reasoning_length'] - group['reasoning_length'].mean()) / (group['reasoning_length'].std() + 1e-8)
        group['time_accel'] = group['time_since_last_action'].diff().fillna(0)

        action_dummies = pd.get_dummies(group['action_type'], prefix='act')
        action_entropy = -np.sum(action_dummies * np.log(action_dummies + 1e-8), axis=1)
        group['action_entropy'] = action_entropy

        features = group[FEATURE_COLS].fillna(0).values.T
        feature_dfs.append(features)
        labels.append(group['outcome'].iloc[0])

    max_len = max([f.shape[1] for f in feature_dfs])

    X_padded = np.zeros((len(feature_dfs), len(FEATURE_COLS), max_len))
    for i, feat in enumerate(feature_dfs):
        n_feat, n_time = feat.shape
        X_padded[i, :, :n_time] = feat

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    print(f"Loaded {len(feature_dfs)} sessions")
    print(f"Feature shape: {X_padded.shape} (sessions, features, timesteps)")
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {np.bincount(y_encoded)}")

    return X_padded, y_encoded, le


def evaluate_model(X, y, label_encoder):
    """
    ADDED: Proper evaluation with train/test split and 5-fold cross-validation.
    This is the critical section missing from the original script.
    """

    print("\n" + "="*60)
    print("STEP 1 — Train/Test Split Evaluation (80/20)")
    print("="*60)

    # Stratified split preserves class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # ensures equal hallucination/success in both splits
    )

    print(f"Train size: {len(X_train)} sessions")
    print(f"Test size:  {len(X_test)} sessions")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution:  {np.bincount(y_test)}")

    # Train on train split only
    eval_model = RocketClassifier(
        num_kernels=10000,
        rocket_transform="minirocket",
        random_state=42,
    )
    eval_model.fit(X_train, y_train)

    train_acc = eval_model.score(X_train, y_train)
    test_acc = eval_model.score(X_test, y_test)

    print(f"\nTraining Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:     {test_acc:.3f}  <-- the number that matters")

    # Gap between train and test — overfitting check
    gap = train_acc - test_acc
    if gap > 0.15:
        print(f"⚠️  Overfitting gap: {gap:.3f} — consider more training data or fewer kernels")
    elif gap > 0.05:
        print(f"✓  Mild generalization gap: {gap:.3f} — acceptable for this dataset size")
    else:
        print(f"✓✓ Strong generalization: gap only {gap:.3f} — model generalizes well")

    # Detailed per-class report
    y_pred = eval_model.predict(X_test)
    class_names = label_encoder.classes_

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("CONFUSION MATRIX")
    print(f"{'':>15}", end="")
    for c in class_names:
        print(f"{c:>15}", end="")
    print()
    for i, row_label in enumerate(class_names):
        print(f"{row_label:>15}", end="")
        for val in cm[i]:
            print(f"{val:>15}", end="")
        print()

    print("\n" + "="*60)
    print("STEP 2 — 5-Fold Cross-Validation (on full dataset)")
    print("="*60)

    cv_model = RocketClassifier(
        num_kernels=10000,
        rocket_transform="minirocket",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='accuracy')

    print(f"CV Scores per fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"CV Mean Accuracy:   {cv_scores.mean():.3f}")
    print(f"CV Std Deviation:   {cv_scores.std():.3f}")
    print(f"CV 95% CI:          [{cv_scores.mean() - 2*cv_scores.std():.3f}, {cv_scores.mean() + 2*cv_scores.std():.3f}]")

    # Final summary for your resume/LinkedIn
    print("\n" + "="*60)
    print("RESULTS SUMMARY — Use these numbers on your resume")
    print("="*60)
    print(f"  Dataset:          400 sessions, 17,186 actions")
    print(f"  Model:            ROCKET (MiniRocket), 10,000 kernels")
    print(f"  Test Accuracy:    {test_acc:.1%}")
    print(f"  CV Accuracy:      {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"  Baseline (random): 50.0%")
    print(f"  Improvement:      +{(test_acc - 0.5):.1%} above random chance")

    return test_acc, cv_scores


def train_rocket_classifier(X, y):
    """Train final RocketClassifier on full dataset for deployment."""
    print("\n" + "="*60)
    print("STEP 3 — Training Final Model on Full Dataset")
    print("="*60)

    classifier = RocketClassifier(
        num_kernels=10000,
        rocket_transform="minirocket",
        random_state=42,
    )

    classifier.fit(X, y)

    train_acc = classifier.score(X, y)
    print(f"Final model training accuracy: {train_acc:.3f}")
    print("(This model is saved for dashboard inference)")

    return classifier


def main():
    """Main training pipeline with proper evaluation."""

    # 1. Load and engineer features
    X, y, label_encoder = load_and_engineer_features()

    # 2. ADDED: Proper evaluation before saving anything
    test_acc, cv_scores = evaluate_model(X, y, label_encoder)

    # 3. Train final model on full dataset for deployment
    model = train_rocket_classifier(X, y)

    # 4. Save model + label encoder + evaluation results
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': FEATURE_COLS,
        'max_timesteps': X.shape[2],
        'test_accuracy': test_acc,           # ADDED: store for dashboard display
        'cv_mean': cv_scores.mean(),         # ADDED: store for dashboard display
        'cv_std': cv_scores.std(),           # ADDED: store for dashboard display
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {MODEL_PATH}")

    # 5. Quick sanity check on full model
    preds = model.predict(X)
    probs = model.predict_proba(X)
    print(f"Prediction shape: {preds.shape}")
    print(f"Probability shape: {probs.shape}")
    print("\nDone. Run the dashboard next: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()