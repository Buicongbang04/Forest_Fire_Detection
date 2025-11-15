import os, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
import joblib

FEATURES = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
TARGET   = "Classes"

def log_init(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    def _log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return _log, log_path

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df[TARGET] = df[TARGET].map(lambda x: 1 if str(x).strip().lower()=="fire" else 0)
    return df

def split_train_val_test(df, test_size=0.2, val_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=random_state)
    val_ratio = val_size / (1.0 - test_size)
    tr_df, val_df = train_test_split(train_df, test_size=val_ratio, stratify=train_df[TARGET], random_state=random_state)
    return tr_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def balance_near_equal(train_df: pd.DataFrame, delta: int = 100, random_state: int = 42) -> pd.DataFrame:
    fire_df = train_df[train_df[TARGET] == 1]
    notfire_df = train_df[train_df[TARGET] == 0]
    n_fire = len(fire_df)
    n_not_target = min(len(notfire_df), n_fire + max(0, int(delta)))
    notfire_keep = notfire_df.sample(n=n_not_target, random_state=random_state, replace=False)
    balanced = pd.concat([fire_df, notfire_keep], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced

def build_scaler(scaler_type: str):
    return StandardScaler() if scaler_type == "standard" else MinMaxScaler()

def build_model(model_type: str, random_state: int, pos_weight: float):
    mt = model_type.lower()
    if mt == "logistic":
        cw = {0: 1.0, 1: float(pos_weight)}
        return LogisticRegression(max_iter=1000, random_state=random_state, class_weight=cw, n_jobs=-1)

    if mt == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=4000,             # tăng số cây
            learning_rate=0.02,            # giảm tốc độ học
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="aucpr",
            scale_pos_weight=float(pos_weight),
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist"
        )

    cw = {0: 1.0, 1: float(pos_weight)}
    return RandomForestClassifier(
        n_estimators=800, max_depth=None, min_samples_leaf=2,
        class_weight=cw, n_jobs=-1, random_state=random_state
    )

def get_prob_pos(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X).reshape(-1,1)
    return MinMaxScaler().fit_transform(scores).ravel()

def tune_threshold_for_precision(y_true, prob_pos, min_precision=0.30):
    thresholds = np.unique(np.concatenate(([0.0, 1.0], prob_pos)))
    thresholds.sort()

    best = None
    fallback = None

    def score_tuple(diff, balanced, precision):
        return (diff, -balanced, -precision)

    for thr in thresholds:
        pred = (prob_pos >= thr).astype(int)
        fire_precision = precision_score(y_true, pred, pos_label=1, zero_division=0)
        fire_recall = recall_score(y_true, pred, pos_label=1, zero_division=0)
        not_fire_recall = recall_score(y_true, pred, pos_label=0, zero_division=0)
        balanced = 0.5 * (fire_recall + not_fire_recall)
        diff = abs(fire_recall - not_fire_recall)
        candidate = (score_tuple(diff, balanced, fire_precision), thr, fire_precision, fire_recall, not_fire_recall)

        if fallback is None or candidate[0] < fallback[0]:
            fallback = candidate
        if fire_precision < min_precision:
            continue
        if best is None or candidate[0] < best[0]:
            best = candidate

    chosen = best if best is not None else fallback

    if chosen is None:
        return 0.5, 0.0, 0.0, 0.0

    _, thr, fire_precision, fire_recall, not_fire_recall = chosen
    return float(thr), float(fire_precision), float(fire_recall), float(not_fire_recall)

def plot_pr_roc_cm(y_true, prob_pos, y_pred, run_dir: Path, title: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    prec, rec, _ = precision_recall_curve(y_true, prob_pos)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall: {title}")
    plt.tight_layout(); plt.savefig(run_dir / "pr_curve.png", dpi=160); plt.close()

    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC: {title}"); plt.legend()
    plt.tight_layout(); plt.savefig(run_dir / "roc_curve.png", dpi=160); plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues"); plt.colorbar(im)
    plt.xticks([0,1], ["not fire","fire"]); plt.yticks([0,1], ["not fire","fire"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=9)
    plt.title(f"Confusion Matrix: {title}")
    plt.tight_layout(); plt.savefig(run_dir / "confusion_matrix.png", dpi=160); plt.close()

def compute_metrics(y_true, prob_pos, y_pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, prob_pos)),
        "pr_auc": float(average_precision_score(y_true, prob_pos)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "fire_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fire_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }

def train_one_run(csv_path: str, model_type: str, scaler_type: str, test_size: float, val_size: float, balance_delta: int, min_precision: float, random_state: int, base_log_dir: str):
    run_dir = Path(base_log_dir) / f"{model_type}_{scaler_type}"
    log, log_path = log_init(run_dir)
    t0 = time.time()

    log(f"==== RUN: {model_type.upper()} × {scaler_type.upper()} ====")
    log(f"csv_path={csv_path}")
    log(f"test_size={test_size}  val_size={val_size}")
    log(f"balance_delta={balance_delta}  min_precision={min_precision}")
    log(f"random_state={random_state}")
    log("-"*58)

    df = load_dataset(Path(csv_path))
    tr_df, val_df, te_df = split_train_val_test(df, test_size=test_size, val_size=val_size, random_state=random_state)
    log(f"Train size (raw): {len(tr_df):,} | Val: {len(val_df):,} | Test: {len(te_df):,}")

    tr_df = balance_near_equal(tr_df, delta=balance_delta, random_state=random_state)
    log(f"Balanced Train → fire={int((tr_df[TARGET]==1).sum()):,}, not_fire={int((tr_df[TARGET]==0).sum()):,}")

    n_pos = int((tr_df[TARGET]==1).sum())
    n_neg = int((tr_df[TARGET]==0).sum())
    pos_weight = (n_neg / max(1, n_pos))

    scaler = build_scaler(scaler_type)
    X_tr, y_tr = tr_df[FEATURES].values, tr_df[TARGET].values
    X_val, y_val = val_df[FEATURES].values, val_df[TARGET].values
    X_te,  y_te  = te_df[FEATURES].values,  te_df[TARGET].values
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    model = build_model(model_type, random_state, pos_weight)
    log("Training...")

    if isinstance(model, XGBClassifier) and XGBClassifier is not None:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=200,
            verbose=False
        )
    else:
        model.fit(X_tr, y_tr)

    val_prob = get_prob_pos(model, X_val)
    best_th, fire_precision, fire_recall, not_fire_recall = tune_threshold_for_precision(
        y_val, val_prob, min_precision=min_precision
    )
    balance_gap = abs(fire_recall - not_fire_recall)
    log(
        "Chosen threshold={:.4f} (fire_precision≈{:.3f}, fire_recall≈{:.3f}, "
        "not_fire_recall≈{:.3f}, gap≈{:.3f})".format(
            best_th, fire_precision, fire_recall, not_fire_recall, balance_gap
        )
    )

    te_prob = get_prob_pos(model, X_te)
    te_pred = (te_prob >= best_th).astype(int)

    report = classification_report(y_te, te_pred, target_names=["not fire","fire"], digits=3)
    cm = confusion_matrix(y_te, te_pred)
    metrics = compute_metrics(y_te, te_prob, te_pred)

    log("\nClassification Report:\n" + report)
    log(f"Confusion Matrix:\n{cm}")
    log(f"Extra Metrics: {json.dumps(metrics, indent=2)}")

    joblib.dump(model, run_dir / "fire_model.pkl")
    joblib.dump(scaler, run_dir / "fire_scaler.pkl")
    with open(run_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "csv": csv_path,
            "model": model_type,
            "scaler": scaler_type,
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
            "balance_delta": balance_delta,
            "min_precision": min_precision,
            "pos_weight_used": pos_weight,
            "train_time_sec": round(time.time()-t0, 2),
            "log_path": str(log_path)
        }, f, indent=2, ensure_ascii=False)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_pr_roc_cm(y_te, te_prob, te_pred, run_dir, title=f"{model_type}_{scaler_type}")
    log(f"Saved artifacts & charts to: {run_dir}")
    log("="*58)

def main():
    CONFIG = {
        "csv_path": "data/data/clean/clean_2000_2024.csv",
        "test_size": 0.10,
        "val_size": 0.20,
        "random_state": 42,
        "balance_delta": 0,
        "min_precision": 0.18,
        "MODELS":  ["randomforest", "logistic", "xgboost"],
        "SCALERS": ["standard", "minmax"],
        "log_dir": "logs"
    }
    for m in CONFIG["MODELS"]:
        for s in CONFIG["SCALERS"]:
            train_one_run(
                csv_path     = CONFIG["csv_path"],
                model_type   = m,
                scaler_type  = s,
                test_size    = CONFIG["test_size"],
                val_size     = CONFIG["val_size"],
                balance_delta= CONFIG["balance_delta"],
                min_precision= CONFIG["min_precision"],
                random_state = CONFIG["random_state"],
                base_log_dir = CONFIG["log_dir"],
            )

if __name__ == "__main__":
    main()
