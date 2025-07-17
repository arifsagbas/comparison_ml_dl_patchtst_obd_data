import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from patchtst_model import PatchTST
from training_utils import train_model, evaluate_model, plot_confusion_matrix, plot_roc_curve

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

base_path = r"\\OBD"
window_lengths = [3, 5, 7, 9]
feature_sets = {
    'ECU': list(range(15, 20)),
    'ECU_MOT': list(range(0, 20)),
    'ECU_MOT_OVER': list(range(0, 24)),
    'ALL': list(range(0, 37))
}
class_map = {'AGGRESSIVE': 0, 'CALM': 1, 'NORMAL': 2}
class_labels = ["AGGRESSIVE", "CALM", "NORMAL"]

for win in window_lengths:
    for fset_name, feature_indices in feature_sets.items():
        set_seed(42)
        result_dir = f"results_patchtst/{win}s_{fset_name}/"
        os.makedirs(result_dir, exist_ok=True)

        X, y = [], []
        for cls in class_map:
            label = class_map[cls]
            folder_path = os.path.join(base_path, f"{cls}_{win}s")
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(folder_path, file), header=None, sep=';')
                    if df.shape[1] < max(feature_indices) + 1:
                        continue
                    X.append(df.iloc[:, feature_indices].values)
                    y.append(label)

        X = np.array(X)
        y = np.array(y)
        print(f"[{win}s - {fset_name}] Total: {len(X)}, Shape: {X.shape}")

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_reports = []
        y_true_all, y_pred_all, y_prob_all = [], [], []
        fold_train_loss, fold_val_loss = [], []
        fold_histories = []

        

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
            print(f"--- Fold {fold} ---")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val, dtype=torch.long))

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)

            model = PatchTST(input_channels=X.shape[2], seq_len=X.shape[1], num_classes=3, patch_len=15).cuda()
            model, history = train_model(model, train_loader, val_loader, epochs=100, patience=10)

            fold_histories.append(history)


            y_pred, y_prob = evaluate_model(model, val_loader)

            report_dict = classification_report(y_val, y_pred, output_dict=True, target_names=class_labels)
            fold_reports.append(pd.DataFrame(report_dict).T)

            y_true_all.extend(y_val)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_prob)

            fold_train_loss.append(history['train_loss'])
            fold_val_loss.append(history['val_loss'])

            plot_confusion_matrix(y_val, y_pred, title=f"Confusion Matrix - Fold {fold}",
                                  save_path=os.path.join(result_dir, f"cm_fold{fold}.png"))
            plot_roc_curve(y_val, y_prob, n_classes=3,
                           title=f"ROC Curve - Fold {fold}",
                           save_path=os.path.join(result_dir, f"roc_fold{fold}.png"))

        mean_report_df = pd.concat(fold_reports).groupby(level=0).mean()
        mean_report_df = mean_report_df[["precision", "recall", "f1-score", "support"]]
        mean_report_df.index.name = "class"
        mean_report_df.to_csv(os.path.join(result_dir, "final_classification_report.csv"), float_format="%.4f")

        ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_all, display_labels=class_labels)
        plt.title("Confusion Matrix (All Folds)")
        plt.savefig(os.path.join(result_dir, "cm_all_folds.png"))
        plt.close()

        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true_all, classes=[0, 1, 2])
        y_prob_all = np.array(y_prob_all)
        plt.figure()
        for i, cls in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_all[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob_all[:, i])
            plt.plot(fpr, tpr, label=f"{cls} (AUC = {auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (All Folds)")
        plt.legend()
        plt.savefig(os.path.join(result_dir, "roc_all_folds.png"))
        plt.close()

        max_len = max(len(l) for l in fold_train_loss)
        train_matrix = np.array([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in fold_train_loss])
        val_matrix = np.array([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in fold_val_loss])

        train_mean = np.nanmean(train_matrix, axis=0)
        val_mean = np.nanmean(val_matrix, axis=0)
        train_std = np.nanstd(train_matrix, axis=0)
        val_std = np.nanstd(val_matrix, axis=0)

        plt.figure()
        plt.plot(train_mean, label="Train Loss")
        plt.fill_between(range(max_len), train_mean-train_std, train_mean+train_std, alpha=0.3)
        plt.plot(val_mean, label="Val Loss")
        plt.fill_between(range(max_len), val_mean-val_std, val_mean+val_std, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Mean Train/Val Loss (±std)")
        plt.legend()
        plt.savefig(os.path.join(result_dir, "loss_all_folds.png"))
        plt.close()

# Accuracy grafiklerini çiz
        fold_train_acc = [h["train_acc"] for h in fold_histories]
        fold_val_acc = [h["val_acc"] for h in fold_histories]

        max_acc_len = max(len(l) for l in fold_train_acc)
        train_acc_matrix = np.array([np.pad(l, (0, max_acc_len - len(l)), constant_values=np.nan) for l in fold_train_acc])
        val_acc_matrix = np.array([np.pad(l, (0, max_acc_len - len(l)), constant_values=np.nan) for l in fold_val_acc])

        train_acc_mean = np.nanmean(train_acc_matrix, axis=0)
        val_acc_mean = np.nanmean(val_acc_matrix, axis=0)
        train_acc_std = np.nanstd(train_acc_matrix, axis=0)
        val_acc_std = np.nanstd(val_acc_matrix, axis=0)

        plt.figure()
        plt.plot(train_acc_mean, label="Train Accuracy")
        plt.fill_between(range(max_acc_len), train_acc_mean-train_acc_std, train_acc_mean+train_acc_std, alpha=0.3)
        plt.plot(val_acc_mean, label="Val Accuracy")
        plt.fill_between(range(max_acc_len), val_acc_mean-val_acc_std, val_acc_mean+val_acc_std, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Mean Train/Val Accuracy (±std)")
        plt.legend()
        plt.savefig(os.path.join(result_dir, "acc_all_folds.png"))
        plt.close()
