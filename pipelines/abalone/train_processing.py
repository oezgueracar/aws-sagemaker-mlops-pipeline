# train_processing.py
# Trains a model inside a Processing job and writes model.tar.gz
# Need to do it this way since free tier aws does not support training jobs

import argparse, os, tarfile, subprocess, sys, pandas as pd

def pip_install(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--validation", type=str, required=False)
    p.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    p.add_argument("--num-round", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.2)
    return p.parse_args()

def load_xy(folder, filename):
    df = pd.read_csv(os.path.join(folder, filename), header=None)  # NO headers
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, y

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    pip_install("xgboost")
    import xgboost as xgb

    X_train, y_train = load_xy(args.train, "train.csv")
    dtrain = xgb.DMatrix(X_train, label=y_train)

    evals = [(dtrain, "train")]
    if args.validation:
        X_val, y_val = load_xy(args.validation, "validation.csv")
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "validation"))

    params = {"objective": "reg:squarederror", "max_depth": args.max_depth, "eta": args.eta, "verbosity": 1}
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=args.num_round, evals=evals)

    # Save in SageMaker XGBoost serving format: model.tar.gz containing 'xgboost-model'
    raw_model_path = os.path.join(args.model_dir, "xgboost-model")
    bst.save_model(raw_model_path)
    tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(raw_model_path, arcname="xgboost-model")

if __name__ == "__main__":
    main()