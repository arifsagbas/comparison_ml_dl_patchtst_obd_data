import os
import numpy as np
import torch
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from utils import set_seed, load_data_from_folders, save_results, plot_metrics
from models import CNNModel, LSTMModel, GRUModel, RNNModel
from trainer import train_and_evaluate

# === CONFIG ===
base_dir = r"\OBD"
window_lengths = ['3s', '5s', '7s', '9s']
class_folders = ['AGGRESSIVE', 'CALM', 'NORMAL']
feature_modes = {
    'ECU': list(range(15, 20)),
    'ECU+Motion': list(range(0, 15)) + list(range(15, 20)),
    'ECU+Motion+Overpass': list(range(0, 15)) + list(range(15, 20)) + list(range(20, 24)),
    'All': list(range(0, 37))
}
model_types = {
    'CNN': CNNModel,
    'LSTM': LSTMModel,
    'GRU': GRUModel,
    #'ViT': ViTModel
    'RNN': RNNModel
}
num_folds = 10
batch_size = 8
epochs = 100
seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === RUN ===
set_seed(seed)

for window in window_lengths:
    for feat_name, feat_idx in feature_modes.items():
        # 1. Veri setini yükle (np.array X, y)
        X, y = load_data_from_folders(base_dir, class_folders, window, feat_idx)
        print(f"[{window} - {feat_name}] Shape: {X.shape}")

        # 2. Fold bazlı eğitim
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for model_name, model_fn in model_types.items():
            print(f"▶ Training {model_name} on {window} - {feat_name}")
            results = train_and_evaluate(
                X, y,
                model_fn=model_fn,
                input_dim=X.shape[2],
                num_classes=len(np.unique(y)),
                folds=skf,
                batch_size=batch_size,
                epochs=epochs,
                model_name=model_name,
                dataset_name=f"{window}_{feat_name}",
                device=device,
                results_dir=os.path.join(base_dir, 'results_dl')
            )

            # 3. Kaydet
            save_results(results, model_name, window, feat_name, base_dir)
            plot_metrics(results, model_name, window, feat_name, base_dir)

print("✅ Tüm deep learning deneyleri tamamlandı.")
