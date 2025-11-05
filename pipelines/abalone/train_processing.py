# train_processing.py
# Trains a model inside a Processing job and writes model.tar.gz
# Need to do it this way since free tier aws does not support training jobs

import argparse, os, tarfile, subprocess, sys
import pandas as pd
import numpy as np

def ensure_pkg(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--validation", type=str, required=False)
    p.add_argument("--label-col", type=str, default="rings")     # Abalone label
    p.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    p.add_argument("--num-round", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.2)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # Install xgboost if image doesn't have it
    ensure_pkg("xgboost")
    import xgboost as xgb

    # Load data prepared by preprocess step
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    if args.validation:
        val_df = pd.read_csv(os.path.join(args.validation, "validation.csv"))

    y_train = train_df[args.label-col].values if hasattr(args, "label-col") else train_df[args.label_col].values
    X_train = train_df.drop(columns=[args.label_col]).values

    if args.validation:
        y_val = val_df[args.label_col].values
        X_val = val_df.drop(columns=[args.label_col]).values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val) if args.validation else None

    params = {
        "objective": "reg:squarederror",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "verbosity": 1,
    }
    evals = [(dtrain, "train")]
    if dval is not None:
        evals.append((dval, "validation"))

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=evals
    )

    # Save model in SageMaker XGBoost serving format
    # The XGBoost serving container expects a file named 'xgboost-model' inside model.tar.gz
    raw_model_path = os.path.join(args.model_dir, "xgboost-model")
    bst.save_model(raw_model_path)

    tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(raw_model_path, arcname="xgboost-model")

if __name__ == "__main__":
    main()