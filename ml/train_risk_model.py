import os
import math
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "rba-dataset.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "app", "ml")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "adaptive_risk_model.joblib")


def load_rba_csv(path: str, target_normal: int = 15000, target_attack: int = 3000, chunksize: int = 50000, max_chunks: int = 20):
    """
    Streams the large RBA CSV in chunks using chunksize:
    - Retains attack & ATO records (minority class).
    - Collects up to target_normal baseline logins.
    """
    print(f"📖 Streaming from: {path}")
    attack_chunks = []
    normal_chunks = []
    normal_count = 0
    attack_count = 0

    # Read in chunks to avoid memory overflow
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk.columns = chunk.columns.str.strip()

        # Parse boolean targets reliably (handles strings or raw booleans)
        is_ato = chunk['Is Account Takeover'].astype(str).str.lower().isin(['true', '1', 't'])
        is_attack_ip = chunk['Is Attack IP'].astype(str).str.lower().isin(['true', '1', 't'])
        is_attack = is_ato | is_attack_ip

        if attack_count < target_attack:
            attacks = chunk[is_attack]
            needed_attacks = target_attack - attack_count
            sample_att = attacks.head(needed_attacks)
            attack_chunks.append(sample_att)
            attack_count += len(sample_att)

        # 2. Grab normal rows up to target
        if normal_count < target_normal:
            normals = chunk[~is_attack]
            needed_norm = target_normal - normal_count
            sample_norm = normals.head(needed_norm)
            normal_chunks.append(sample_norm)
            normal_count += len(sample_norm)

        if normal_count >= target_normal and attack_count >= target_attack:
            break

    
    df_attacks = pd.concat(attack_chunks, ignore_index=True) if attack_chunks else pd.DataFrame()
    df_normal = pd.concat(normal_chunks, ignore_index=True) if normal_chunks else pd.DataFrame()
    df = pd.concat([df_normal, df_attacks], ignore_index=True)  

    # Feature Engineering
    y = ((df['Is Account Takeover'].astype(str).str.lower().isin(['true', '1', 't'])) |
         (df['Is Attack IP'].astype(str).str.lower().isin(['true', '1', 't']))).astype(int)

    # Convert epoch timestamp to hours and day of week
    ts = pd.to_datetime(df['Login Timestamp'], unit='s', errors='coerce')
    hours = ts.dt.hour.fillna(12)

    # Clean OS families
    def clean_os(val):
        val = str(val).lower()
        if 'android' in val: return 'Android'
        if 'ios' in val or 'iphone' in val: return 'iOS'
        if 'windows' in val: return 'Windows'
        if 'linux' in val: return 'Linux'
        if 'mac' in val: return 'MacOS'
        return 'Other'

    X = pd.DataFrame({
        'device_type': df['Device Type'].fillna('unknown').astype(str).str.lower(),
        'os_family': df['OS Name and Version'].apply(clean_os),
        'hour_sin': np.sin(2 * np.pi * hours / 24.0),
        'hour_cos': np.cos(2 * np.pi * hours / 24.0),
        'rtt_ms': pd.to_numeric(df['Round-Trip Time [ms]'], errors='coerce').clip(1, 4000).fillna(100.0),
    })

    return X, y


def generate_fallback_rba_dataset(n_samples=10000):
    """Fallback generator if no CSV exists."""
    print("ℹ️  Synthesizing 10,000 RBA-calibrated records for training...")
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        is_attack = np.random.rand() < 0.12
        if not is_attack:
            device_type = 'mobile'
            os_family = np.random.choice(['Android', 'iOS'], p=[0.75, 0.25])
            country = np.random.choice(['KE', 'UG', 'TZ'], p=[0.90, 0.06, 0.04])
            hour = int(np.clip(np.random.normal(13, 4), 6, 22))
            day_of_week = np.random.randint(0, 7)
            rtt_ms = np.random.normal(60, 20)
            label = 0
        else:
            device_type = np.random.choice(['mobile', 'desktop', 'bot'], p=[0.2, 0.3, 0.5])
            os_family = np.random.choice(['Linux', 'Windows', 'Other', 'Android'], p=[0.4, 0.3, 0.2, 0.1])
            country = np.random.choice(['US', 'RU', 'CN', 'DE', 'UNKNOWN'], p=[0.3, 0.2, 0.2, 0.15, 0.15])
            hour = np.random.choice([0, 1, 2, 3, 4, 23])
            day_of_week = np.random.randint(0, 7)
            rtt_ms = np.random.normal(320, 110)
            label = 1

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        data.append({
            'device_type': device_type,
            'os_family': os_family,
            'country': country,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_of_week': day_of_week,
            'rtt_ms': max(1.0, rtt_ms),
            'label': label,
        })

    df = pd.DataFrame(data)
    return df.drop(columns=['label']), df['label']


def run_training():
    if os.path.exists(DATA_PATH):
        X, y = load_rba_csv(DATA_PATH)
    else:
        X, y = generate_fallback_rba_dataset(n_samples=10000)

    categorical_cols = ['device_type', 'os_family']
    numerical_cols = ['hour_sin', 'hour_cos', 'rtt_ms']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numerical_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    print("⏳ Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Training Random Forest Adaptive Risk Model...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("               📊 MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN: {cm[0][0]:<6}  FP: {cm[0][1]:<6}]")
    print(f"  [FN: {cm[1][0]:<6}  TP: {cm[1][1]:<6}]")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Trusted (0)', 'Anomaly (1)']))

    # Feature Importance (XAI)
    cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([cat_feature_names, numerical_cols])
    importances = pipeline.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("🏆 Top 7 Most Influential Risk Signals:")
    for _, row in importance_df.head(7).iterrows():
        print(f"  • {row['Feature']:<25} : {row['Importance']*100:.2f}%")
    print("=" * 55 + "\n")

    # Export
    joblib.dump(pipeline, OUTPUT_MODEL_PATH)
    print(f"✅ Model successfully exported to:\n   {os.path.abspath(OUTPUT_MODEL_PATH)}")


if __name__ == '__main__':
    run_training()