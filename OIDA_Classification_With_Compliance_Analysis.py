import pandas as pd
import numpy as np
import re, warnings, zipfile, os
from pathlib import Path
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for r in ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'omw-1.4']:
    try: nltk.download(r, quiet=True)
    except: pass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, confusion_matrix)
from sklearn.utils import class_weight
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LSTM, Embedding, Dropout,
                                     SpatialDropout1D, Bidirectional,
                                     BatchNormalization)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ── Colab-safe matplotlib ────────────────────────────────────────────────────
try:
    import google.colab
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

print("=" * 80)
print("  PART 1 : Document Classification")
print("  PART 2 : Compliance Analysis (Step A + B + C)")
print("  PART 3 : Tactic Identification")
print("=" * 80)
print(f"TensorFlow : {tf.__version__}")
print("=" * 80)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHARED UTILITIES                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TextPreprocessor:
    def __init__(self):
        try:    self.stop_words = set(stopwords.words('english'))
        except: self.stop_words = set()
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

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_dataset(filepath):
    print(f"\nLoading : {filepath}")

    def safe_read(file_obj, sep='|'):
        """
        Try multiple encodings to safely read CSV
        """
        for enc in ['utf-8', 'cp1252', 'latin1']:
            try:
                return pd.read_csv(file_obj, sep=sep,
                                   encoding=enc,
                                   on_bad_lines='skip',
                                   low_memory=False)
            except:
                continue

        # Fallback without separator assumption
        for enc in ['utf-8', 'cp1252', 'latin1']:
            try:
                return pd.read_csv(file_obj,
                                   encoding=enc,
                                   on_bad_lines='skip',
                                   low_memory=False)
            except:
                continue

        return None

    # ── HANDLE CSV FILE ────────────────────────────────────────
    df = safe_read(filepath)

    if df is None or df.empty:
        print("❌ Failed to read file with all encoding attempts.")
        return pd.DataFrame()

    # ── FIX INDEX (if weird multi-index occurs) ─────────────────────
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    # ── AUTO-DETECT TEXT COLUMN ─────────────────────────────────────
    text_col = next((c for c in df.columns
                     if any(k in c.lower()
                            for k in ['ocr', 'text', 'content', 'body'])), None)

    df['text'] = df[text_col].astype(str) if text_col else df.iloc[:, 0].astype(str)

    # ── CREATE DOCUMENT ID ──────────────────────────────────────────
    if 'id' in df.columns:
        df['doc_id'] = df['id'].astype(str)
    elif 'bates' in df.columns:
        df['doc_id'] = df['bates'].astype(str)
    else:
        df['doc_id'] = 'doc_' + pd.Series(range(len(df))).astype(str)

    # ── CLEAN DATA ──────────────────────────────────────────────────
    df = (
        df[df['text'].notna()]
        .pipe(lambda d: d[d['text'].str.len() > 50])
        .drop_duplicates(subset=['text'])
        .reset_index(drop=True)
    )

    print(f"   ✓ Loaded {len(df)} documents")
    return df

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LABEL CREATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_labels(df):
    print("\nCreating labels ...")
    promo = ['sales','marketing','promote','promotional','market','physician',
             'prescriber','prescription','revenue','profit','target','campaign',
             'sell','aggressive','quota','incentive','bonus','pressure','goal',
             'performance','distribution','detailing','advertise','commercial']
    sci   = ['study','research','clinical','trial','data','efficacy','safety',
             'patient','dose','treatment','results','findings','analysis',
             'experiment','placebo','randomized','peer-review','hypothesis',
             'methodology','statistical','controlled','cohort','endpoint']
    reg   = ['regulation','regulatory','compliance','legal','fda','dea',
             'government','law','court','litigation','lawsuit','enforcement',
             'approval','attorney','judge','testimony','settlement','deposition',
             'subpoena','investigation','violation','penalty','consent']
    labels = []
    for text in df['text']:
        t = text.lower()
        p = sum(1 for k in promo if k in t)
        s = sum(1 for k in sci   if k in t)
        r = sum(1 for k in reg   if k in t)
        if p == s == r:
            labels.append('scientific' if len(text) > 8000 else 'promotional')
        else:
            labels.append(max({'promotional': p, 'scientific': s, 'regulatory': r},
                              key={'promotional': p, 'scientific': s,
                                   'regulatory': r}.get))
    df['label'] = labels
    print("\n   Label Distribution:")
    for lbl, cnt in df['label'].value_counts().items():
        print(f"   {lbl:15s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")
    return df

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1 — CLASSIFICATION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "="*80)
    print("PART 1 : RANDOM FOREST")
    print("="*80)
    max_feat = min(5000, len(X_train)*2)
    vec = TfidfVectorizer(max_features=max_feat, ngram_range=(1, 3),
                          max_df=0.9, min_df=1, sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    n_est = min(500, max(100, len(X_train)))
    rf = RandomForestClassifier(n_estimators=n_est, class_weight='balanced',
                                random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_tr, y_train)
    pred = rf.predict(X_te)
    print(f"\nAccuracy : {accuracy_score(y_test, pred)*100:.2f}%")
    print(f"F1-Macro : {f1_score(y_test, pred, average='macro'):.4f}")
    print(f"OOB      : {rf.oob_score_:.4f}")
    print("\n" + classification_report(y_test, pred, digits=4))
    return rf, vec, pred

def train_ensemble(X_train, X_test, y_train, y_test, vec):
    print("\n" + "="*80)
    print("PART 1 : ENSEMBLE VOTING")
    print("="*80)
    X_tr = vec.transform(X_train)
    X_te = vec.transform(X_test)
    ens = VotingClassifier(
        estimators=[
            ('rf',  RandomForestClassifier(n_estimators=300,
                                           class_weight='balanced',
                                           random_state=42, n_jobs=-1)),
            ('svm', LinearSVC(class_weight='balanced',
                              random_state=42, max_iter=2000)),
            ('lr',  LogisticRegression(class_weight='balanced',
                                       random_state=42,
                                       max_iter=1000, n_jobs=-1)),
        ], voting='hard', n_jobs=-1)
    ens.fit(X_tr, y_train)
    pred = ens.predict(X_te)
    print(f"\nAccuracy : {accuracy_score(y_test, pred)*100:.2f}%")
    print(f"F1-Macro : {f1_score(y_test, pred, average='macro'):.4f}")
    print("\n" + classification_report(y_test, pred, digits=4))
    return ens, pred

def train_bilstm(X_train, X_test, y_train, y_test, label_encoder):
    print("\n" + "="*80)
    print("PART 1 : BiLSTM")
    print("="*80)
    y_tr_enc = label_encoder.transform(y_train)
    y_te_enc = label_encoder.transform(y_test)
    y_tr_cat = to_categorical(y_tr_enc)
    cw = class_weight.compute_class_weight('balanced',
             classes=np.unique(y_tr_enc), y=y_tr_enc)
    cw_dict = {i: w for i, w in enumerate(cw)}
    max_words = min(15000, len(X_train)*10)
    max_len   = min(400, max(len(str(t).split()) for t in X_train)+50)
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    X_tr = pad_sequences(tok.texts_to_sequences(X_train),
                         maxlen=max_len, padding='post')
    X_te = pad_sequences(tok.texts_to_sequences(X_test),
                         maxlen=max_len, padding='post')
    print(f"\n   Building BiLSTM (vocab={max_words}, seq_len={max_len}) ...")
    model = Sequential([
        Embedding(max_words, 150, input_length=max_len, mask_zero=True),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(128, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.2)),
        BatchNormalization(),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax'),
    ])
    model.compile(optimizer=Adam(0.001, clipnorm=1.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    early  = EarlyStopping(monitor='val_accuracy', patience=5,
                           restore_best_weights=True, verbose=1)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                               patience=3, min_lr=1e-6, verbose=1)
    print("\nTraining BiLSTM ...")
    model.fit(X_tr, y_tr_cat, validation_split=0.15,
              epochs=2,
              batch_size=min(32, len(X_train)//5),
              callbacks=[early, reduce],
              class_weight=cw_dict, verbose=1)
    pred = label_encoder.inverse_transform(
               np.argmax(model.predict(X_te, verbose=0), axis=1))
    print(f"\nBiLSTM Accuracy : {accuracy_score(y_test, pred)*100:.2f}%")
    print(f"BiLSTM F1       : {f1_score(y_test, pred, average='macro'):.4f}")
    print("\n" + classification_report(y_test, pred, digits=4))
    return model, pred

def plot_confusion_matrices(y_test, predictions_dict, labels):
    print("\nPlotting confusion matrices ...")
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]
    for idx, (name, pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=[l.title() for l in labels],
                    yticklabels=[l.title() for l in labels])
        axes[idx].set_title(
            f'{name}\nAcc: {accuracy_score(y_test, pred)*100:.2f}%',
            fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Predicted')
    plt.suptitle('PART 1 — Confusion Matrices',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Saved: confusion_matrices.png")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 2 — COMPLIANCE ANALYSIS                                            ║  
# ╚══════════════════════════════════════════════════════════════════════════╝

def _cluster_documents(df_subset, n_clusters, label_prefix,
                        ngram_range=(1, 3), max_features=3000,
                        top_phrases=8, min_docs=2):
    pre = TextPreprocessor()
    texts   = df_subset['text'].tolist()
    doc_ids = df_subset['doc_id'].tolist()
    cleaned = [pre.clean(t) for t in texts]

    # ── TF-IDF vectorisation ────────────────────────────────────────────────
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=max(2, len(texts)//20),   # at least 5% of docs must have term
        max_df=0.90,
        stop_words='english',
        sublinear_tf=True
    )
    try:
        mat = vec.fit_transform(cleaned)
    except Exception:
        return []

    if mat.shape[1] < 2:
        return []

    feature_names = np.array(vec.get_feature_names_out())

    # ── LSA dimensionality reduction ────────────────────────────────────────
    n_comp = min(50, mat.shape[1]-1, mat.shape[0]-1)
    if n_comp < 2:
        return []
    svd     = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced = svd.fit_transform(mat)

    # ── K-Means clustering ──────────────────────────────────────────────────
    k = min(n_clusters, len(df_subset)//max(min_docs, 1))
    k = max(k, 2)
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(reduced)

    # ── Extract top phrases per cluster ─────────────────────────────────────
    topics = []
    for cid in range(k):
        mask     = labels == cid
        if mask.sum() < min_docs:
            continue

        # Average TF-IDF vector for this cluster → pick top phrases
        cluster_vec   = mat[mask].toarray().mean(axis=0)
        top_idx       = cluster_vec.argsort()[::-1][:top_phrases]
        phrases       = [feature_names[i] for i in top_idx
                         if cluster_vec[i] > 0]
        if not phrases:
            continue

        # Auto label = highest-scoring phrase → title case
        auto_label = phrases[0].title()

        # Collect evidence snippets
        cluster_doc_indices = [i for i, m in enumerate(mask) if m]
        snippets = []
        for di in cluster_doc_indices[:5]:
            raw = texts[di]
            for ph in phrases[:3]:
                idx = raw.lower().find(ph.lower())
                if idx >= 0:
                    snip = raw[max(0,idx-80):min(len(raw),idx+80)].strip()
                    snip = re.sub(r'\s+', ' ', snip)
                    snippets.append(f'...{snip[:160]}...')
                    break

        topics.append({
            'label':          auto_label,
            'top_phrases':    phrases,
            'keywords':       phrases,          # alias — used in cosine matching
            'document_count': int(mask.sum()),
            'percentage':     mask.sum() / len(df_subset) * 100,
            'doc_ids':        [doc_ids[i] for i in cluster_doc_indices[:5]],
            'examples':       snippets[:2],
        })

    # Sort by document count descending
    topics.sort(key=lambda x: x['document_count'], reverse=True)
    return topics

# ── Step A : Discover regulatory topics from regulatory documents ────────────

def extract_regulatory_requirements(df_reg, n_topics=8):
    print("\n" + "="*80)
    print("PART 2 — STEP A : DISCOVERING REGULATORY TOPICS  (Fully Dynamic)")
    print("  Method : TF-IDF (1-3 grams)  →  LSA  →  K-Means clustering")
    print("  No hardcoded categories or keyword lists.")
    print("="*80)

    if len(df_reg) < 3:
        print("  ⚠  Too few regulatory documents.")
        return {}

    n_topics = min(n_topics, len(df_reg)//2)
    topics   = _cluster_documents(
        df_reg, n_clusters=n_topics,
        label_prefix='REG',
        ngram_range=(1, 3),
        max_features=3000,
        top_phrases=8
    )

    reqs = {}
    for t in topics:
        reqs[t['label']] = t

    print(f"\n  Regulatory docs scanned          : {len(df_reg)}")
    print(f"  Regulatory topics AUTO-DISCOVERED: {len(reqs)}\n")

    if reqs:
        print(f"  {'DISCOVERED TOPIC':<35s}  {'DOCS':>5s}  {'%':>6s}  "
              f"KEY PHRASES (auto-extracted)")
        print("  " + "─"*90)
        for name, d in reqs.items():
            bar     = "█" * int(d['percentage']/5)
            phrases = ' | '.join(d['top_phrases'][:3])
            print(f"  {name:<35s}  {d['document_count']:>5d}  "
                  f"{d['percentage']:>5.1f}%  {bar}")
            print(f"  {'':35s}  {'Phrases:':>8s}  \"{phrases}\"")
            print()
    else:
        print("  ⚠  Could not discover regulatory topics from this dataset.")

    return reqs


# ── Step B : Discover company practice topics from promotional documents ─────

def extract_company_practices(df_promo, n_topics=8):
    print("\n" + "="*80)
    print("PART 2 — STEP B : DISCOVERING COMPANY PRACTICES  (Fully Dynamic)")
    print("  Method : TF-IDF (1-3 grams)  →  LSA  →  K-Means clustering")
    print("  No hardcoded categories or keyword lists.")
    print("="*80)

    if len(df_promo) < 3:
        print("  ⚠  Too few promotional documents.")
        return {}

    n_topics = min(n_topics, len(df_promo)//2)
    topics   = _cluster_documents(
        df_promo, n_clusters=n_topics,
        label_prefix='PRAC',
        ngram_range=(1, 3),
        max_features=3000,
        top_phrases=8
    )

    pracs = {}
    for t in topics:
        pracs[t['label']] = t

    print(f"\n  Promotional docs scanned         : {len(df_promo)}")
    print(f"  Company practices AUTO-DISCOVERED: {len(pracs)}\n")

    if pracs:
        print(f"  {'DISCOVERED PRACTICE':<35s}  {'DOCS':>5s}  {'%':>6s}  "
              f"KEY PHRASES (auto-extracted)")
        print("  " + "─"*90)
        for name, d in pracs.items():
            bar     = "█" * int(d['percentage']/5)
            phrases = ' | '.join(d['top_phrases'][:3])
            print(f"  {name:<35s}  {d['document_count']:>5d}  "
                  f"{d['percentage']:>5.1f}%  {bar}")
            print(f"  {'':35s}  {'Phrases:':>8s}  \"{phrases}\"")
            print()
    else:
        print("  ⚠  Could not discover company practices from this dataset.")

    return pracs

# ── Step C : Dynamic compliance mapping (unchanged from v4) ─────────────────

def build_dynamic_mapping(reqs, pracs, threshold=0.10):
    """
    Auto-pairs every discovered regulatory topic with the most
    semantically similar company practice using TF-IDF cosine similarity.
    No hardcoded pairs — fully driven by what Steps A and B found.
    """
    if not reqs or not pracs:
        return {}, {}

    reg_names  = list(reqs.keys())
    prac_names = list(pracs.keys())

    def desc(name, data):
        kw = ' '.join(data.get('keywords', []))
        n  = name.replace('/', ' ').replace('-', ' ')
        return f"{n} {n} {kw}"

    all_desc = ([desc(n, reqs[n])  for n in reg_names] +
                [desc(n, pracs[n]) for n in prac_names])
    tfidf    = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

    try:
        mat = tfidf.fit_transform(all_desc)
    except Exception:
        return {reg: None for reg in reg_names}, {reg: 0.0 for reg in reg_names}

    sim = cosine_similarity(mat[:len(reg_names)], mat[len(reg_names):])

    mapping, scores = {}, {}
    for i, reg in enumerate(reg_names):
        best_idx   = int(np.argmax(sim[i]))
        best_score = sim[i][best_idx]
        mapping[reg] = prac_names[best_idx] if best_score >= threshold else None
        scores[reg]  = round(float(best_score), 4)

    return mapping, scores


def analyze_compliance(reqs, pracs, threshold=0.10):
    print("\n" + "="*80)
    print("PART 2 — STEP C : COMPLIANCE ANALYSIS  (Dynamic Mapping)")
    print("  Each auto-discovered regulation is matched to the most")
    print("  similar auto-discovered company practice via cosine similarity.")
    print("="*80)

    mapping, scores = build_dynamic_mapping(reqs, pracs, threshold)

    # Print the auto-generated mapping
    print(f"\n  AUTO-GENERATED REGULATION → PRACTICE PAIRS:")
    print(f"  {'REGULATION TOPIC':<35s}  {'MATCHED PRACTICE':<35s}  "
          f"{'SCORE':>8s}  STATUS")
    print("  " + "─"*90)
    for reg, prac in mapping.items():
        sc = scores.get(reg, 0.0)
        st = "✔ MATCHED" if prac else "✘ NO MATCH"
        print(f"  {reg:<35s}  {(prac or '—'):<35s}  {sc:>8.4f}  {st}")

    # Run compliance checks
    findings    = []
    comp_ok     = 0
    viol        = 0
    total       = len(mapping)

    print(f"\n\n  COMPLIANCE CHECK RESULTS:")
    print(f"  {'REGULATION':<35s}  {'PRACTICE':<35s}  {'STATUS':<22s}  RISK")
    print("  " + "─"*100)

    for reg, prac in mapping.items():
        if prac is None:
            sev, status = "🟢 NONE", "✓  COMPLIANT (no match)"
            comp_ok += 1
            rd, pd_ = reqs.get(reg, {}).get('document_count', 0), 0
        else:
            has_r = reg  in reqs
            has_p = prac in pracs
            if has_r and has_p:
                pct = pracs[prac]['percentage']
                sev = ("🔴 HIGH"   if pct > 20 else
                       "🟠 MEDIUM" if pct > 10 else "🟡 LOW")
                status = "⚠  POTENTIAL VIOLATION"
                viol  += 1
            elif has_r and not has_p:
                sev, status = "🟢 NONE", "✓  COMPLIANT"
                comp_ok += 1
            else:
                sev, status = "—", "— NOT APPLICABLE"
                comp_ok += 1
                total   -= 1
            rd  = reqs.get(reg,  {}).get('document_count', 0)
            pd_ = pracs.get(prac, {}).get('document_count', 0)

        findings.append({
            'regulation': reg,
            'practice':   prac or '—',
            'severity':   sev,
            'status':     status,
            'reg_docs':   rd,
            'prac_docs':  pd_,
            'sim_score':  scores.get(reg, 0.0),
        })
        print(f"  {reg:<35s}  {(prac or '—'):<35s}  "
              f"{status:<22s}  {sev}")

    cr = comp_ok / max(total, 1) * 100
    print("\n  " + "─"*100)
    print(f"\n  ✦ COMPLIANCE SCORE  : {cr:.1f}%  "
          f"({comp_ok} compliant / {total} checks)")
    print(f"  ✦ VIOLATIONS FOUND  : {viol}")
    return findings, cr, mapping, scores


# ── Part 2 detailed printout ─────────────────────────────────────────────────

def print_part2_detailed(reqs, pracs, findings, compliance_rate):
    violations = [f for f in findings if 'VIOLATION' in f['status']]
    compliant  = [f for f in findings if 'COMPLIANT' in f['status']]

    print("\n\n" + "█"*80)
    print("█" + " "*25 + "PART 2  DETAILED FINDINGS REPORT" + " "*21 + "█")
    print("█"*80)

    # Section 1 — Regulatory topics
    print("\n" + "─"*80)
    print("  SECTION 1 ▸ REGULATORY TOPICS AUTO-DISCOVERED FROM DATA")
    print("─"*80)
    print("  (These were NOT predefined — the algorithm found them by itself)\n")
    for i, (cat, d) in enumerate(reqs.items(), 1):
        print(f"  [{i}] TOPIC : {cat.upper()}")
        print(f"       Docs   : {d['document_count']}  ({d['percentage']:.1f}%)")
        print(f"       Phrases: {' | '.join(d['top_phrases'][:5])}")
        if d['examples']:
            print(f"       Evidence snippet:")
            print(f"         \"{d['examples'][0]}\"")
        print()

    # Section 2 — Company practices
    print("─"*80)
    print("  SECTION 2 ▸ COMPANY PRACTICES AUTO-DISCOVERED FROM DATA")
    print("─"*80)
    print("  (These were NOT predefined — the algorithm found them by itself)\n")
    for i, (cat, d) in enumerate(pracs.items(), 1):
        print(f"  [{i}] PRACTICE : {cat.upper()}")
        print(f"       Docs     : {d['document_count']}  ({d['percentage']:.1f}%)")
        print(f"       Phrases  : {' | '.join(d['top_phrases'][:5])}")
        if d['examples']:
            print(f"       Evidence snippet:")
            print(f"         \"{d['examples'][0]}\"")
        print()

    # Section 3 — Conflicts
    print("─"*80)
    print("  SECTION 3 ▸ COMPLIANCE CONFLICTS")
    print("─"*80)
    if violations:
        for v in violations:
            re_ = reqs.get(v['regulation'],  {}).get('examples', [])
            pe_ = pracs.get(v['practice'],   {}).get('examples', [])
            rph = reqs.get(v['regulation'],  {}).get('top_phrases', [])
            pph = pracs.get(v['practice'],   {}).get('top_phrases', [])
            print(f"  {v['severity']}  ───────────────────────────────────────────")
            print(f"  REGULATION TOPIC : {v['regulation']}")
            print(f"  COMPANY PRACTICE : {v['practice']}")
            print(f"  SIMILARITY SCORE : {v['sim_score']:.4f}  "
                  f"(auto-matched)")
            print(f"  REG KEY PHRASES  : {' | '.join(rph[:4])}")
            print(f"  PRAC KEY PHRASES : {' | '.join(pph[:4])}")
            print(f"  REG EVIDENCE     : Found in {v['reg_docs']} regulatory doc(s).")
            if re_: print(f"    ↳ {re_[0]}")
            print(f"  PRAC EVIDENCE    : Found in {v['prac_docs']} promotional doc(s).")
            if pe_: print(f"    ↳ {pe_[0]}")
            print(f"  PLAIN ENGLISH    : The regulation cluster '{v['regulation']}'")
            print(f"                     conflicts with the company practice cluster")
            print(f"                     '{v['practice']}' — potential evasion.")
            print()
    else:
        print("  ✓ No conflicts detected.\n")

    if compliant:
        print("  COMPLIANT AREAS:")
        for c in compliant:
            print(f"    ✓  {c['regulation']}  →  no conflicting practice found")
        print()

    # Section 4 — Scorecard
    print("─"*80)
    bf  = int(compliance_rate/5)
    bar = "█"*bf + "░"*(20-bf)
    print(f"\n  Compliance  : [{bar}]  {compliance_rate:.1f}%")
    lvl = ("🟢 GOOD"     if compliance_rate >= 75 else
           "🟠 MODERATE" if compliance_rate >= 50 else "🔴 CRITICAL")
    print(f"  Status      : {lvl}")
    print(f"  Violations  : {len(violations)}")
    print(f"  Compliant   : {len(compliant)}")
    print("\n" + "█"*80)


def plot_part2_dashboard(reqs, pracs, findings, compliance_rate, dynamic_mapping):
    print("\nGenerating Part 2 dashboard ...")
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("PART 2 — Compliance Dashboard  (100% Dynamic — No Hardcoded Categories)",
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.50, wspace=0.40)

    # Panel A — Regulatory topics
    ax_a = fig.add_subplot(gs[0, 0])
    if reqs:
        cats  = list(reqs.keys())
        vals  = [reqs[c]['document_count'] for c in cats]
        short = [c[:22] for c in cats]
        bars_ = ax_a.barh(short, vals, color='steelblue', edgecolor='white')
        ax_a.bar_label(bars_, padding=3, fontsize=8)
        ax_a.invert_yaxis()
        ax_a.set_xlabel("Documents")
        ax_a.set_title("A — Regulatory Topics\n(auto-discovered)",
                       fontsize=10, fontweight='bold')

    # Panel B — Company practices
    ax_b = fig.add_subplot(gs[0, 1])
    if pracs:
        cats   = list(pracs.keys())
        colors = ['tomato'  if pracs[c]['percentage'] > 20 else
                  'orange'  if pracs[c]['percentage'] > 10 else 'gold'
                  for c in cats]
        short  = [c[:22] for c in cats]
        bars_ = ax_b.barh(short, [pracs[c]['document_count'] for c in cats],
                          color=colors, edgecolor='white')
        ax_b.bar_label(bars_, padding=3, fontsize=8)
        ax_b.invert_yaxis()
        ax_b.set_xlabel("Documents")
        ax_b.set_title("B — Company Practices\n(auto-discovered; "
                        "red>20%, orange>10%, gold<10%)",
                       fontsize=10, fontweight='bold')
        ax_b.legend(handles=[
            mpatches.Patch(color='tomato', label='High >20%'),
            mpatches.Patch(color='orange', label='Med 10-20%'),
            mpatches.Patch(color='gold',   label='Low <10%')], fontsize=7)

    # Panel C — Conflict heatmap
    ax_c = fig.add_subplot(gs[1, 0])
    regs   = list(dynamic_mapping.keys())
    matrix = np.zeros((len(regs), 2))
    for i, (r, p) in enumerate(dynamic_mapping.items()):
        matrix[i, 0] = 1 if r in reqs else 0
        matrix[i, 1] = 1 if (p and p in pracs) else 0
    ax_c.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(['Regulation\nExists', 'Practice\nFound'], fontsize=8)
    ax_c.set_yticks(range(len(regs)))
    ax_c.set_yticklabels([r[:25] for r in regs], fontsize=7)
    ax_c.set_title("C — Conflict Heatmap  (both red = ⚠ conflict)",
                   fontsize=10, fontweight='bold')
    for i in range(len(regs)):
        for j in range(2):
            ax_c.text(j, i, "YES" if matrix[i, j] == 1 else "NO",
                      ha='center', va='center', fontsize=8,
                      fontweight='bold', color='white')

    # Panel D — Scorecard
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    violations = [f for f in findings if 'VIOLATION' in f['status']]
    compliant  = [f for f in findings if 'COMPLIANT' in f['status']]
    items_ = [
        ("COMPLIANCE SCORE",  f"{compliance_rate:.1f}%",
         'green' if compliance_rate >= 75 else
         'orange' if compliance_rate >= 50 else 'red'),
        ("REG TOPICS FOUND",  str(len(reqs)),         'steelblue'),
        ("PRACTICES FOUND",   str(len(pracs)),         'steelblue'),
        ("VIOLATIONS",        str(len(violations)),   'red' if violations else 'green'),
        ("COMPLIANT AREAS",   str(len(compliant)),    'green'),
    ]
    ax_d.set_title("D — Scorecard", fontsize=10, fontweight='bold')
    y = 0.92
    for lbl, val, col in items_:
        ax_d.text(0.05, y, lbl+":", fontsize=10, va='top', transform=ax_d.transAxes)
        ax_d.text(0.72, y, val, fontsize=12, va='top', fontweight='bold',
                  transform=ax_d.transAxes, color=col)
        y -= 0.17

    plt.savefig('part2_compliance_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   ✓ Saved: part2_compliance_dashboard.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 3 — TACTIC IDENTIFICATION  (unchanged from v4)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def discover_tactics_from_text(df_promo, n_tactics=10, top_ngrams=8):
    print("\n" + "="*80)
    print("PART 3 — STEP A : UNSUPERVISED TACTIC DISCOVERY")
    print("  Method : TF-IDF bigrams/trigrams  +  K-Means clustering")
    print("="*80)

    if len(df_promo) < 3:
        print("  ⚠  Too few documents.")
        return []

    pre     = TextPreprocessor()
    texts   = df_promo['text'].tolist()
    doc_ids = df_promo['doc_id'].tolist()
    cleaned = [pre.clean(t) for t in texts]

    vec = TfidfVectorizer(ngram_range=(2, 3), max_features=2000,
                          min_df=2, max_df=0.85,
                          stop_words='english', sublinear_tf=True)
    try:
        tfidf_mat = vec.fit_transform(cleaned)
    except Exception:
        print("  ⚠  Not enough vocabulary.")
        return []

    feature_names = np.array(vec.get_feature_names_out())
    n_comp = min(50, tfidf_mat.shape[1]-1, tfidf_mat.shape[0]-1)
    if n_comp < 2:
        return []

    svd      = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced  = svd.fit_transform(tfidf_mat)
    n_clust  = max(2, min(n_tactics, len(df_promo)//3, 10))
    km       = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
    labels_k = km.fit_predict(reduced)

    print(f"\n  Documents analysed  : {len(df_promo)}")
    print(f"  Tactics discovered  : {n_clust}\n")

    tactics = []
    for cid in range(n_clust):
        mask         = labels_k == cid
        cluster_docs = [i for i, m in enumerate(mask) if m]
        if not cluster_docs:
            continue
        cv         = tfidf_mat[mask].toarray().mean(axis=0)
        top_idx    = cv.argsort()[::-1][:top_ngrams]
        top_phrases = [feature_names[i] for i in top_idx if cv[i] > 0]
        if not top_phrases:
            continue

        label    = top_phrases[0].title()
        evidence = []
        for di in cluster_docs[:5]:
            raw = texts[di]
            try:    sents = sent_tokenize(raw)
            except: sents = raw.split('.')
            for ph in top_phrases[:3]:
                for sent in sents:
                    if ph.lower() in sent.lower() and len(sent.strip()) > 30:
                        evidence.append(re.sub(r'\s+', ' ', sent.strip())[:200])
                        break
                if evidence:
                    break

        tactics.append({
            'tactic_id':          cid + 1,
            'label':              label,
            'top_phrases':        top_phrases,
            'doc_count':          int(mask.sum()),
            'doc_ids':            [doc_ids[i] for i in cluster_docs[:5]],
            'evidence_sentences': list(set(evidence))[:3],
            'percentage':         mask.sum()/len(df_promo)*100,
        })

    tactics.sort(key=lambda x: x['doc_count'], reverse=True)
    return tactics

def classify_tactic_intent(tactics, reqs):
    if not tactics:
        return tactics

    sales_vocab = (
        "increase sales revenue profit market share growth target customers "
        "expand distribution boost performance quota campaign promote sell "
        "physician prescriber doctor detailing commercial advertising"
    )
    evasion_vocab = (
        "avoid bypass circumvent hide conceal mislead loophole indirect "
        "workaround evade reframe disguise alternative framing rebranding "
        "claim safer less harmful reduced risk"
    )

    reg_names = list(reqs.keys())
    reg_texts = [' '.join(reqs[r].get('keywords', reqs[r].get('top_phrases', [])))
                 + ' ' + r for r in reg_names]
    corpus    = [sales_vocab, evasion_vocab] + reg_texts

    intent_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        intent_mat = intent_vec.fit_transform(corpus)
    except Exception:
        return tactics

    for tac in tactics:
        tactic_text = ' '.join(tac['top_phrases'])
        try:
            tv   = intent_vec.transform([tactic_text])
            sims = cosine_similarity(tv, intent_mat)[0]
        except Exception:
            tac['intent']            = 'UNKNOWN'
            tac['intent_scores']     = {}
            tac['regulation_evaded'] = 'None identified'
            continue

        sales_score   = float(sims[0])
        evasion_score = float(sims[1])
        reg_sims      = {reg_names[i]: float(sims[2+i])
                         for i in range(len(reg_names))}

        if sales_score > evasion_score and sales_score > 0.05:
            intent = "SALES BOOSTING"
        elif evasion_score > sales_score and evasion_score > 0.05:
            intent = "REGULATION EVASION"
        elif sales_score > 0.03 or evasion_score > 0.03:
            intent = "MIXED  (Sales + Evasion)"
        else:
            intent = "GENERAL PRACTICE"

        best_reg   = max(reg_sims, key=reg_sims.get) if reg_sims else None
        best_score = reg_sims.get(best_reg, 0.0)
        reg_evaded = (best_reg if best_score > 0.05
                      else "No specific regulation matched")

        tac['intent']            = intent
        tac['intent_scores']     = {'sales_similarity':   round(sales_score, 4),
                                    'evasion_similarity':  round(evasion_score, 4)}
        tac['regulation_evaded'] = reg_evaded
        tac['reg_evasion_score'] = round(best_score, 4)

    return tactics


def attribute_tactics_to_documents(df_promo, tactics):
    if not tactics:
        return []
    doc_profiles = []
    for _, row in df_promo.iterrows():
        tl = row['text'].lower()
        dt = []
        for tac in tactics:
            mp = [p for p in tac['top_phrases'] if p.lower() in tl]
            if mp:
                dt.append({'tactic_id':       tac['tactic_id'],
                           'tactic_label':    tac['label'],
                           'intent':          tac.get('intent', 'UNKNOWN'),
                           'matched_phrases': mp[:3],
                           'regulation_evaded': tac.get('regulation_evaded',
                                                         'None identified')})
        if dt:
            doc_profiles.append({'doc_id':      row['doc_id'],
                                  'text_snippet': row['text'][:200].replace('\n',' '),
                                  'tactics':      dt,
                                  'tactic_count': len(dt)})
    return doc_profiles

def generate_insights(tactics, doc_profiles):
    insights = []
    for tac in tactics:
        intent    = tac.get('intent', 'UNKNOWN')
        reg_evaded = tac.get('regulation_evaded', 'None identified')
        pct        = tac.get('percentage', 0)
        doc_count  = tac.get('doc_count', 0)
        phrases    = tac.get('top_phrases', [])

        if 'SALES' in intent:
            goal       = "Increase product sales and expand market reach"
            risk_level = "🟠 MEDIUM — Sales practices may breach marketing guidelines"
        elif 'EVASION' in intent:
            goal       = "Circumvent or downplay regulatory restrictions"
            risk_level = "🔴 HIGH — Tactics appear designed to evade specific regulations"
        elif 'MIXED' in intent:
            goal       = "Simultaneously drive sales while minimising regulatory exposure"
            risk_level = "🔴 HIGH — Combined sales and evasion signals detected"
        else:
            goal       = "General business practice"
            risk_level = "🟡 LOW — No strong sales or evasion signal detected"

        phrase_sample = ', '.join(f'"{p}"' for p in phrases[:3])
        if reg_evaded != "No specific regulation matched":
            plain = (f"Documents using phrases like {phrase_sample} suggest the company "
                     f"is '{goal.lower()}'. This may conflict with the auto-discovered "
                     f"regulation cluster '{reg_evaded}'. "
                     f"Found in {doc_count} documents ({pct:.1f}%).")
        else:
            plain = (f"Documents using phrases like {phrase_sample} indicate "
                     f"'{goal.lower()}'. No direct regulatory conflict identified. "
                     f"Present in {doc_count} documents ({pct:.1f}%).")

        insights.append({
            'tactic_id':          tac['tactic_id'],
            'tactic_name':        tac['label'],
            'key_phrases':        phrases[:5],
            'goal':               goal,
            'intent':             intent,
            'regulation_evaded':  reg_evaded,
            'documents_affected': doc_count,
            'percentage':         pct,
            'risk_level':         risk_level,
            'plain_english':      plain,
        })
    return insights

def print_part3_output(tactics, doc_profiles, insights):
    print("\n\n" + "▓"*80)
    print("▓" + " "*20 + "PART 3  —  TACTIC IDENTIFICATION REPORT" + " "*19 + "▓")
    print("▓"*80)

    # Section A — overview table
    print("\n" + "═"*80)
    print("  SECTION A ▸ TACTICS DISCOVERED  (Data-Driven, No Hardcoding)")
    print("═"*80)
    print(f"  Total unique tactics found : {len(tactics)}\n")
    print(f"  {'#':<4s}  {'TACTIC LABEL':<28s}  {'DOCS':>5s}  {'%':>6s}  "
          f"{'INTENT':<25s}  REGULATION EVADED")
    print("  " + "─"*100)
    for t in tactics:
        reg = t.get('regulation_evaded', '—')
        if len(reg) > 28: reg = reg[:25]+"..."
        print(f"  {t['tactic_id']:<4d}  {t['label']:<28s}  "
              f"{t['doc_count']:>5d}  {t['percentage']:>5.1f}%  "
              f"{t.get('intent','UNKNOWN'):<25s}  {reg}")

    # Section B — detailed profiles
    print("\n\n" + "═"*80)
    print("  SECTION B ▸ DETAILED TACTIC PROFILES")
    print("═"*80)
    for t in tactics:
        intent = t.get('intent', 'UNKNOWN')
        icon   = ("🔴" if "EVASION" in intent or "MIXED" in intent
                  else "🟠" if "SALES" in intent else "🟡")
        print(f"  {icon}  TACTIC #{t['tactic_id']} : {t['label'].upper()}")
        print(f"  {'─'*76}")
        print(f"  Documents  : {t['doc_count']}  ({t['percentage']:.1f}%)")
        print(f"  Intent     : {intent}")
        print(f"  Goal       : {t.get('goal','—')}")
        print(f"  Reg. Evaded: {t.get('regulation_evaded','None identified')}")
        sc = t.get('intent_scores', {})
        if sc:
            print(f"  Scores     : Sales={sc.get('sales_similarity',0):.4f}  "
                  f"Evasion={sc.get('evasion_similarity',0):.4f}  "
                  f"RegMatch={t.get('reg_evasion_score',0):.4f}")
        print(f"\n  Key Phrases (auto-extracted from documents):")
        for i, ph in enumerate(t['top_phrases'][:6], 1):
            print(f"    {i}. \"{ph}\"")
        ev = t.get('evidence_sentences', [])
        if ev:
            print(f"\n  Real Evidence from Documents:")
            for i, s in enumerate(ev[:2], 1):
                print(f"    [{i}] \"{s}\"")
        print()

    # Section C — insights
    print("═"*80)
    print("  SECTION C ▸ GENERATED INSIGHTS")
    print("═"*80)
    for ins in insights:
        print(f"  ┌─ INSIGHT #{ins['tactic_id']} : {ins['tactic_name'].upper()}")
        print(f"  │  Goal            : {ins['goal']}")
        print(f"  │  Intent          : {ins['intent']}")
        print(f"  │  Regulation Risk : {ins['regulation_evaded']}")
        print(f"  │  Risk Level      : {ins['risk_level']}")
        print(f"  │  Affected Docs   : {ins['documents_affected']}  "
              f"({ins['percentage']:.1f}%)")
        print(f"  │")
        print(f"  │  Plain English :")
        words, line = ins['plain_english'].split(), "  │    "
        for w in words:
            if len(line)+len(w)+1 > 76:
                print(line); line = "  │    "+w+" "
            else:
                line += w+" "
        if line.strip(): print(line)
        print(f"  └{'─'*77}")
        print()

    # Section D — per-document attribution
    print("═"*80)
    print("  SECTION D ▸ PER-DOCUMENT TACTIC ATTRIBUTION  (Top 10 docs)")
    print("═"*80)
    for dp in sorted(doc_profiles, key=lambda x: x['tactic_count'],
                     reverse=True)[:10]:
        print(f"  📄 Doc ID : {dp['doc_id']}")
        print(f"     Preview : \"{dp['text_snippet'][:120]}...\"")
        print(f"     Tactics ({dp['tactic_count']}):")
        for tac in dp['tactics']:
            ph = ', '.join(f'"{p}"' for p in tac['matched_phrases'])
            print(f"       • [{tac['intent']:<22s}]  {tac['tactic_label']}")
            print(f"         Phrases : {ph}")
            print(f"         Risk    : {tac['regulation_evaded']}")
        print()

    # Section E — scorecard
    print("═"*80)
    print("  SECTION E ▸ PART 3 SUMMARY SCORECARD")
    print("═"*80)
    sales_n   = sum(1 for t in tactics if 'SALES'   in t.get('intent',''))
    evasion_n = sum(1 for t in tactics if 'EVASION' in t.get('intent','')
                                          or 'MIXED' in t.get('intent',''))
    print(f"\n  Total tactics identified     : {len(tactics)}")
    print(f"  Sales-boosting tactics       : {sales_n}")
    print(f"  Regulation-evasion tactics   : {evasion_n}")
    print(f"  Docs with tactics found      : {len(doc_profiles)}")
    print(f"\n  TACTIC TYPE BREAKDOWN:")
    for intent_type, cnt in Counter(
            t.get('intent','UNKNOWN') for t in tactics).most_common():
        print(f"    {intent_type:<30s} : {cnt:>3d}  {'█'*cnt}")
    print("\n" + "▓"*80)


def plot_part3_dashboard(tactics, insights, doc_profiles):
    if not tactics:
        return
    print("\nGenerating Part 3 dashboard ...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("PART 3 — Tactic Identification Dashboard  (Fully Dynamic)",
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.40)

    # A — frequency
    ax_a = fig.add_subplot(gs[0, 0])
    cols  = ['#d62728' if 'EVASION' in t.get('intent','') or
             'MIXED' in t.get('intent','')
             else '#ff7f0e' if 'SALES' in t.get('intent','')
             else '#1f77b4' for t in tactics]
    labs  = [f"T{t['tactic_id']}: {t['label'][:16]}" for t in tactics]
    bars  = ax_a.barh(labs, [t['doc_count'] for t in tactics],
                      color=cols, edgecolor='white')
    ax_a.bar_label(bars, padding=3, fontsize=8)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Documents")
    ax_a.set_title("A — Tactic Frequency\n(red=evasion, orange=sales, blue=other)",
                   fontsize=10, fontweight='bold')
    ax_a.legend(handles=[
        mpatches.Patch(color='#d62728', label='Evasion/Mixed'),
        mpatches.Patch(color='#ff7f0e', label='Sales Boosting'),
        mpatches.Patch(color='#1f77b4', label='General')], fontsize=7)

    # B — intent pie
    ax_b = fig.add_subplot(gs[0, 1])
    tc   = Counter(t.get('intent','UNKNOWN') for t in tactics)
    ax_b.pie(list(tc.values()),
             labels=[k[:20] for k in tc.keys()],
             autopct='%1.0f%%',
             colors=['#d62728','#ff7f0e','#2ca02c','#1f77b4','#9467bd'],
             startangle=90)
    ax_b.set_title("B — Intent Distribution", fontsize=10, fontweight='bold')

    # C — regulation evaded
    ax_c = fig.add_subplot(gs[0, 2])
    rc = Counter(t.get('regulation_evaded','—') for t in tactics
                 if t.get('regulation_evaded','—') !=
                 'No specific regulation matched')
    if rc:
        bars_c = ax_c.barh([k[:22] for k in rc.keys()],
                            list(rc.values()),
                            color='#d62728', edgecolor='white')
        ax_c.bar_label(bars_c, padding=3, fontsize=8)
        ax_c.invert_yaxis()
    else:
        ax_c.text(0.5, 0.5, "No evasion\ndetected",
                  ha='center', va='center', fontsize=12)
    ax_c.set_xlabel("Tactics")
    ax_c.set_title("C — Regulations Being Evaded",
                   fontsize=10, fontweight='bold')

    # D — histogram
    ax_d = fig.add_subplot(gs[1, 0])
    if doc_profiles:
        cnts = [dp['tactic_count'] for dp in doc_profiles]
        ax_d.hist(cnts, bins=max(set(cnts)), color='steelblue', edgecolor='white')
        ax_d.set_xlabel("Tactics per Document")
        ax_d.set_ylabel("Documents")
        ax_d.set_title("D — Tactics per Document", fontsize=10, fontweight='bold')

    # E — scatter sales vs evasion
    ax_e = fig.add_subplot(gs[1, 1])
    xs = [t.get('intent_scores',{}).get('sales_similarity',   0) for t in tactics]
    ys = [t.get('intent_scores',{}).get('evasion_similarity', 0) for t in tactics]
    ax_e.scatter(xs, ys, c=cols, s=120, edgecolors='white',
                 linewidths=0.5, zorder=3)
    ax_e.axhline(0.05, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_e.axvline(0.05, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    for i, t in enumerate(tactics):
        ax_e.annotate(f"T{t['tactic_id']}", (xs[i], ys[i]),
                      fontsize=7, ha='left', va='bottom')
    ax_e.set_xlabel("Sales Similarity")
    ax_e.set_ylabel("Evasion Similarity")
    ax_e.set_title("E — Sales vs Evasion Score\n(top-right = both)",
                   fontsize=10, fontweight='bold')

    # F — scorecard
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis('off')
    sn = sum(1 for t in tactics if 'SALES'   in t.get('intent',''))
    en = sum(1 for t in tactics if 'EVASION' in t.get('intent','')
                                    or 'MIXED' in t.get('intent',''))
    items = [("TACTICS DISCOVERED", str(len(tactics)),       'black'),
             ("SALES-BOOSTING",     str(sn),                 '#ff7f0e'),
             ("REGULATION-EVASION", str(en),                 '#d62728'),
             ("GENERAL PRACTICE",   str(len(tactics)-sn-en), '#1f77b4'),
             ("DOCS WITH TACTICS",  str(len(doc_profiles)),  'purple')]
    ax_f.set_title("F — Scorecard", fontsize=10, fontweight='bold')
    y = 0.90
    for lbl, val, col in items:
        ax_f.text(0.05, y, lbl+":", fontsize=10, va='top', transform=ax_f.transAxes)
        ax_f.text(0.75, y, val, fontsize=13, va='top', fontweight='bold',
                  transform=ax_f.transAxes, color=col)
        y -= 0.17

    plt.savefig('part3_tactic_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   ✓ Saved: part3_tactic_dashboard.png")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FULL REPORT SAVE                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_full_report(clf_results, reqs, pracs, findings, compliance_rate,
                     tactics, insights, doc_profiles,
                     output_file='full_pipeline_report.txt'):
    lines = []
    sep = "="*80
    def h(t):  lines.extend([sep, t, sep, ""])
    def s(t):  lines.extend(["", t, "-"*len(t)])

    h("OIDA FULL PIPELINE REPORT  v5.0  —  100% DYNAMIC")

    s("PART 1 — CLASSIFICATION RESULTS")
    lines.append(f"  {'Model':<22s}  {'Accuracy':>10s}  {'F1':>8s}")
    lines.append("  "+"-"*44)
    for n, a, f in clf_results:
        lines.append(f"  {n:<22s}  {a*100:>9.2f}%  {f:>8.4f}")

    s("PART 2A — REGULATORY TOPICS  (auto-discovered)")
    for cat, d in reqs.items():
        lines.append(f"\n  {cat}  ({d['document_count']} docs, {d['percentage']:.1f}%)")
        lines.append(f"    Phrases: {' | '.join(d['top_phrases'][:5])}")
        if d['examples']: lines.append(f"    e.g. {d['examples'][0]}")

    s("PART 2B — COMPANY PRACTICES  (auto-discovered)")
    for cat, d in pracs.items():
        lines.append(f"\n  {cat}  ({d['document_count']} docs, {d['percentage']:.1f}%)")
        lines.append(f"    Phrases: {' | '.join(d['top_phrases'][:5])}")
        if d['examples']: lines.append(f"    e.g. {d['examples'][0]}")

    s("PART 2C — COMPLIANCE ANALYSIS")
    lines.append(f"\n  Score : {compliance_rate:.1f}%  |  "
                 f"Violations : {sum(1 for f in findings if 'VIOLATION' in f['status'])}")
    for f in findings:
        lines.append(f"  {f['regulation']:<30s} → {f['practice']:<30s}  "
                     f"{f['status']:<22s}  {f['severity']}")

    s("PART 3 — TACTIC IDENTIFICATION")
    for ins in insights:
        lines.extend([
            f"\n  TACTIC #{ins['tactic_id']} : {ins['tactic_name']}",
            f"  Goal    : {ins['goal']}",
            f"  Intent  : {ins['intent']}",
            f"  Evades  : {ins['regulation_evaded']}",
            f"  Risk    : {ins['risk_level']}",
            f"  Docs    : {ins['documents_affected']}  ({ins['percentage']:.1f}%)",
            f"  Phrases : {', '.join(ins['key_phrases'][:4])}",
            f"  Summary : {ins['plain_english']}",
        ])

    s("PER-DOCUMENT ATTRIBUTION  (top 15)")
    for dp in sorted(doc_profiles, key=lambda x: x['tactic_count'],
                     reverse=True)[:15]:
        lines.append(f"\n  Doc {dp['doc_id']}  ({dp['tactic_count']} tactics)")
        for tac in dp['tactics']:
            lines.append(f"    • {tac['tactic_label']}"
                         f"  [{tac['intent']}]"
                         f"  → {tac['regulation_evaded']}")

    lines.extend(["", sep, "END OF REPORT", sep])
    text = "\n".join(lines)
    try:
        with open(output_file, 'w', encoding='utf-8') as f: f.write(text)
        print(f"\n   ✓ Report saved : {output_file}")
    except:
        fb = '/tmp/'+output_file
        with open(fb, 'w', encoding='utf-8') as f: f.write(text)
        print(f"\n   ✓ Report saved : {fb}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main(filepath):
    print("\n" + "="*80)
    print("STARTING FULL PIPELINE  v5.0  —  100% DYNAMIC")
    print("  PART 1 : Classification  (RF / Ensemble / BiLSTM)")
    print("  PART 2 : Compliance Analysis  (Steps A+B+C — all dynamic)")
    print("  PART 3 : Tactic Identification  (fully dynamic)")
    print("="*80)

    df = load_dataset(filepath)
    if len(df) < 20:
        print(f"\n⚠  Only {len(df)} documents. Need ≥ 20."); return

    df     = create_labels(df)
    counts = df['label'].value_counts()
    valid  = counts[counts >= 3].index
    df     = df[df['label'].isin(valid)].reset_index(drop=True)
    print(f"\n✓ {len(df)} documents  |  {len(valid)} classes")

    print("\nPreprocessing ...")
    pre = TextPreprocessor()
    df['processed'] = df['text'].apply(pre.clean)
    df = df[df['processed'] != 'empty'].reset_index(drop=True)

    test_size = min(0.25, max(0.15, 15/len(df)))
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label'],
        test_size=test_size, random_state=42, stratify=df['label'])
    print(f"   Train : {len(X_train)}  |  Test : {len(X_test)}")

    le = LabelEncoder()
    le.fit(df['label'])

    # ── PART 1 ──────────────────────────────────────────────────────────────
    preds, clf_results = {}, []

    rf, vec, rp = train_random_forest(X_train, X_test, y_train, y_test)
    preds['Random Forest'] = rp
    clf_results.append(('Random Forest',
                        accuracy_score(y_test, rp),
                        f1_score(y_test, rp, average='macro')))

    if len(X_train) >= 50:
        _, ep = train_ensemble(X_train, X_test, y_train, y_test, vec)
        preds['Ensemble'] = ep
        clf_results.append(('Ensemble',
                            accuracy_score(y_test, ep),
                            f1_score(y_test, ep, average='macro')))

    if len(X_train) >= 100:
        _, bp = train_bilstm(X_train, X_test, y_train, y_test, le)
        preds['BiLSTM'] = bp
        clf_results.append(('BiLSTM',
                            accuracy_score(y_test, bp),
                            f1_score(y_test, bp, average='macro')))

    plot_confusion_matrices(y_test, preds, sorted(df['label'].unique()))

    print("\n" + "="*80)
    print("PART 1 COMPLETE")
    print("="*80)
    print(f"\n   {'Model':<22s}  {'Accuracy':>10s}  {'F1-Macro':>10s}")
    print("   " + "-"*46)
    for n, a, f in clf_results:
        print(f"   {n:<22s}  {a*100:>9.2f}%  {f:>10.4f}")

    # ── PART 2 ──────────────────────────────────────────────────────────────
    df_promo = df[df['label'] == 'promotional'].copy()
    df_reg   = df[df['label'] == 'regulatory' ].copy()
    df_sci   = df[df['label'] == 'scientific' ].copy()

    print("\n\n" + "="*80)
    print("STARTING PART 2 — COMPLIANCE ANALYSIS  (100% Dynamic)")
    print("="*80)
    print(f"\n  Promotional : {len(df_promo)}")
    print(f"  Regulatory  : {len(df_reg)}")
    print(f"  Scientific  : {len(df_sci)}")

    if len(df_reg)   < 3: print("\n  ⚠  Few regulatory documents.")
    if len(df_promo) < 3: print("\n  ⚠  Few promotional documents.")

    # Steps A & B — fully dynamic discovery (no hardcoded patterns)
    reqs  = extract_regulatory_requirements(df_reg,   n_topics=8)
    pracs = extract_company_practices(df_promo,        n_topics=8)

    # Step C — dynamic mapping + compliance check
    THRESHOLD = 0.10
    findings, compliance_rate, dyn_map, _ = analyze_compliance(
        reqs, pracs, threshold=THRESHOLD)

    print_part2_detailed(reqs, pracs, findings, compliance_rate)
    plot_part2_dashboard(reqs, pracs, findings, compliance_rate, dyn_map)

    # ── PART 3 ──────────────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("STARTING PART 3 — TACTIC IDENTIFICATION  (Fully Dynamic)")
    print("="*80)

    tactics = discover_tactics_from_text(df_promo, n_tactics=10, top_ngrams=8)

    if tactics:
        tactics      = classify_tactic_intent(tactics, reqs)
        for t in tactics:
            intent = t.get('intent', 'UNKNOWN')
            t['goal'] = ("Increase product sales and expand market reach"
                         if 'SALES' in intent else
                         "Circumvent or downplay regulatory restrictions"
                         if 'EVASION' in intent or 'MIXED' in intent else
                         "General business practice")

        doc_profiles = attribute_tactics_to_documents(df_promo, tactics)
        insights     = generate_insights(tactics, doc_profiles)

        print_part3_output(tactics, doc_profiles, insights)
        plot_part3_dashboard(tactics, insights, doc_profiles)
    else:
        print("\n  ⚠  Not enough data for tactic discovery.")
        doc_profiles, insights = [], []

    # ── SAVE ────────────────────────────────────────────────────────────────
    save_full_report(clf_results, reqs, pracs, findings, compliance_rate,
                     tactics if tactics else [], insights, doc_profiles)

    # ── FINAL SUMMARY ───────────────────────────────────────────────────────
    best_acc  = max(a for _, a, _ in clf_results)
    sales_n   = sum(1 for t in (tactics or []) if 'SALES'   in t.get('intent',''))
    evasion_n = sum(1 for t in (tactics or []) if 'EVASION' in t.get('intent','')
                                                   or 'MIXED' in t.get('intent',''))
    print("\n" + "="*80)
    print("FULL PIPELINE v5.0 COMPLETE  —  100% DYNAMIC")
    print("="*80)
    print(f"\n  PART 1 — Best Accuracy         : {best_acc*100:.2f}%")
    print(f"  PART 2 — Compliance Rate        : {compliance_rate:.1f}%")
    print(f"           Violations Found       : "
          f"{sum(1 for f in findings if 'VIOLATION' in f['status'])}")
    print(f"  PART 3 — Tactics Discovered     : {len(tactics or [])}")
    print(f"           Sales-Boosting         : {sales_n}")
    print(f"           Regulation-Evasion     : {evasion_n}")
    print(f"           Docs With Tactics      : {len(doc_profiles)}")
    print("\n  Saved files :")
    print("    confusion_matrices.png         — Part 1")
    print("    part2_compliance_dashboard.png — Part 2")
    print("    part3_tactic_dashboard.png     — Part 3")
    print("    full_pipeline_report.txt       — Complete report")
    print("="*80)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    try:
        from google.colab import files
        print("\nUpload your CSV or ZIP file:")
        uploaded = files.upload()
        filepath = list(uploaded.keys())[0]
    except ImportError:
        filepath = input("Enter file path: ").strip()
    main(filepath)