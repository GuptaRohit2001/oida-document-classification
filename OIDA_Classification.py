import pandas as pd
import numpy as np
import re, warnings, zipfile, os
from pathlib import Path
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for r in ['punkt','stopwords','wordnet','punkt_tab','omw-1.4']:
    try: nltk.download(r, quiet=True)
    except: pass

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("FAST COMPETITION-WINNING PIPELINE")
print("="*80)
print(f"TensorFlow: {tf.__version__}")
print("="*80)

# ==========================================================
# TEXT PREPROCESSING
# ==========================================================

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        try:
            text = str(text).lower()
            text = re.sub(r'http\S+|www\S+', ' ', text)
            text = re.sub(r'\S+@\S+', ' ', text)
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens
                     if w not in self.stop_words and len(w) > 2]

            return ' '.join(tokens) if tokens else 'empty'
        except:
            return 'empty'

# ==========================================================
# DATA LOADING
# ==========================================================

def load_dataset(filepath):
    print(f"\nLoading: {filepath}")

    if filepath.endswith('.zip'):
        print("Extracting ZIP...")
        with zipfile.ZipFile(filepath, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            dfs = []
            for csv_file in csv_files:
                try:
                    with z.open(csv_file) as f:
                        df_temp = pd.read_csv(f, sep='|', on_bad_lines='skip', low_memory=False)
                        dfs.append(df_temp)
                except:
                    try:
                        with z.open(csv_file) as f:
                            df_temp = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
                            dfs.append(df_temp)
                    except:
                        continue
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        try:
            df = pd.read_csv(filepath, sep='|', on_bad_lines='skip', low_memory=False)
        except:
            df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    text_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ['ocr','text','content','body']):
            text_col = col
            break

    df['text'] = df[text_col].astype(str) if text_col else df.iloc[:, 0].astype(str)

    if 'id' in df.columns:
        df['doc_id'] = df['id'].astype(str)
    elif 'bates' in df.columns:
        df['doc_id'] = df['bates'].astype(str)
    else:
        df['doc_id'] = 'doc_' + pd.Series(range(len(df))).astype(str)

    df = df[df['text'].notna()].copy()
    df = df[df['text'].str.len() > 50].copy()
    df = df.drop_duplicates(subset=['text']).copy()
    df = df.reset_index(drop=True)

    print(f"   ✓ Loaded {len(df)} documents")
    return df

# ==========================================================
# LABEL CREATION
# ==========================================================

def create_labels(df):
    print("\nCreating labels...")

    promo = ['sales','marketing','promote','promotional','market','physician',
             'prescriber','prescription','revenue','profit','target','campaign',
             'sell','aggressive','quota','incentive','bonus','pressure','goal',
             'performance','distribution','detailing','advertise','commercial']

    sci = ['study','research','clinical','trial','data','efficacy','safety',
           'patient','dose','treatment','results','findings','analysis',
           'experiment','placebo','randomized','peer-review','hypothesis',
           'methodology','statistical','controlled','cohort','endpoint']

    reg = ['regulation','regulatory','compliance','legal','fda','dea',
           'government','law','court','litigation','lawsuit','enforcement',
           'approval','attorney','judge','testimony','settlement','deposition',
           'subpoena','investigation','violation','penalty','consent']

    labels = []
    for text in df['text']:
        t = text.lower()
        p = sum(1 for k in promo if k in t)
        s = sum(1 for k in sci if k in t)
        r = sum(1 for k in reg if k in t)

        if p == s == r:
            labels.append('scientific' if len(text) > 8000 else 'promotional')
        else:
            labels.append(max({'promotional':p,'scientific':s,'regulatory':r},
                            key={'promotional':p,'scientific':s,'regulatory':r}.get))

    df['label'] = labels

    print("\n   Label Distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"   {label:15s}: {count:5d} ({count/len(df)*100:.1f}%)")

    return df

# ==========================================================
# TRAIN RANDOM FOREST
# ==========================================================

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "="*80)
    print("RANDOM FOREST")
    print("="*80)

    max_feat = min(5000, len(X_train) * 2)
    vec = TfidfVectorizer(max_features=max_feat, ngram_range=(1,3), max_df=0.9, min_df=1, sublinear_tf=True)

    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    n_est = min(500, max(100, len(X_train)))
    rf = RandomForestClassifier(n_estimators=n_est, class_weight='balanced',
                               random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_tr, y_train)
    rf_pred = rf.predict(X_te)

    acc = accuracy_score(y_test, rf_pred)
    f1 = f1_score(y_test, rf_pred, average='macro')

    print(f"\nAccuracy: {acc*100:.2f}%")
    print(f"F1-Macro: {f1:.4f}")
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print("\n" + classification_report(y_test, rf_pred, digits=4))

    return rf, vec, rf_pred

# ==========================================================
# TRAIN ENSEMBLE
# ==========================================================

def train_ensemble(X_train, X_test, y_train, y_test, vec):
    print("\n" + "="*80)
    print("ENSEMBLE VOTING ")
    print("="*80)

    X_tr = vec.transform(X_train)
    X_te = vec.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
    svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, n_jobs=-1)

    ensemble = VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('lr', lr)], voting='hard', n_jobs=-1)
    ensemble.fit(X_tr, y_train)

    ens_pred = ensemble.predict(X_te)
    acc = accuracy_score(y_test, ens_pred)
    f1 = f1_score(y_test, ens_pred, average='macro')

    print(f"\nEnsemble Accuracy: {acc*100:.2f}%")
    print(f"Ensemble F1: {f1:.4f}")
    print("\n" + classification_report(y_test, ens_pred, digits=4))

    return ensemble, ens_pred

# ==========================================================
# TRAIN BILSTM
# ==========================================================

def train_bilstm(X_train, X_test, y_train, y_test, label_encoder):
    print("\n" + "="*80)
    print("BiLSTM ")
    print("="*80)

    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    y_train_cat = to_categorical(y_train_enc)
    y_test_cat = to_categorical(y_test_enc)

    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
    cw_dict = {i: w for i, w in enumerate(cw)}

    max_words = min(15000, len(X_train) * 10)
    max_len = min(400, max(len(str(t).split()) for t in X_train) + 50)

    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(X_train)

    X_tr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=max_len, padding='post')
    X_te = pad_sequences(tok.texts_to_sequences(X_test), maxlen=max_len, padding='post')

    print(f"\n   Building BiLSTM (vocab={max_words}, len={max_len})...")

    model = Sequential([
        Embedding(max_words, 150, input_length=max_len, mask_zero=True),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        BatchNormalization(),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer=Adam(0.001, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])

    early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    print("\nTraining BiLSTM...")
    history = model.fit(X_tr, y_train_cat, validation_split=0.15, epochs=2,
                       batch_size=min(32, len(X_train)//5), callbacks=[early, reduce],
                       class_weight=cw_dict, verbose=1)

    y_pred = label_encoder.inverse_transform(np.argmax(model.predict(X_te, verbose=0), axis=1))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\nBiLSTM Accuracy: {acc*100:.2f}%")
    print(f"BiLSTM F1: {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, digits=4))

    return model, y_pred

# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_results(y_test, predictions_dict, labels):
    print("\nCreating visualizations...")

    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=[l.title() for l in labels],
                   yticklabels=[l.title() for l in labels])

        acc = accuracy_score(y_test, pred)
        axes[idx].set_title(f'{name}\nAcc: {acc*100:.2f}%', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.show()
    print("   ✓ Saved: confusion_matrices.png")

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main(filepath):
    
    df = load_dataset(filepath)
    if len(df) < 20:
        print(f"\nOnly {len(df)} documents. Need at least 20.")
        return

    df = create_labels(df)

    counts = df['label'].value_counts()
    valid = counts[counts >= 3].index
    df = df[df['label'].isin(valid)].copy()

    print(f"\n✓ Working with {len(df)} documents, {len(valid)} classes")

    print("\nPreprocessing...")
    pre = TextPreprocessor()
    df['processed'] = df['text'].apply(pre.clean)
    df = df[df['processed'] != 'empty'].copy()

    test_size = min(0.25, max(0.15, 15/len(df)))
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label'], test_size=test_size,
        random_state=42, stratify=df['label'])

    print(f"\n   Train: {len(X_train)}, Test: {len(X_test)}")

    label_encoder = LabelEncoder()
    label_encoder.fit(df['label'])

    predictions = {}

    rf, vec, rf_pred = train_random_forest(X_train, X_test, y_train, y_test)
    predictions['Random Forest'] = rf_pred

    if len(X_train) >= 50:
        _, ens_pred = train_ensemble(X_train, X_test, y_train, y_test, vec)
        predictions['Ensemble'] = ens_pred

    if len(X_train) >= 100:
        _, bilstm_pred = train_bilstm(X_train, X_test, y_train, y_test, label_encoder)
        predictions['BiLSTM'] = bilstm_pred

    plot_results(y_test, predictions, sorted(df['label'].unique()))

    print("\n" + "="*80)
    print("FINAL RESULTS (FAST PIPELINE) ")
    print("="*80)
    print(f"\n   {'Model':<20s} {'Accuracy':>10s} {'F1-Macro':>10s}")
    print("   " + "-"*42)

    for name, pred in predictions.items():
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')
        print(f"   {name:<20s} {acc*100:>9.2f}% {f1:>10.4f}")

    print("\n" + "="*80)
    print("FAST PIPELINE COMPLETE! (DistilBERT removed for speed) ")
    print("="*80)

if __name__ == "__main__":
    try:
        from google.colab import files
        print("\nUpload CSV/ZIP:")
        uploaded = files.upload()
        filepath = list(uploaded.keys())[0]
    except:
        filepath = input("File path: ")

    main(filepath)