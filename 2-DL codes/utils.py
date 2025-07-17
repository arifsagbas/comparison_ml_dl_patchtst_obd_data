import os
import torch
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def load_data_from_folders(base_dir, class_folders, window, feature_indices):
    X = []
    y = []
    label_map = {'AGGRESSIVE': 0, 'NORMAL': 1, 'CALM': 2}
    for cls in class_folders:
        folder_name = f"{cls}_{window}"
        folder_path = os.path.join(base_dir, folder_name)
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for file in files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=None, sep=';')  
            if df.shape[1] < max(feature_indices) + 1:
                print(f"⚠️ SKIPPED: {file_path} has only {df.shape[1]} columns, expected at least {max(feature_indices)+1}")
                continue
            try:
                X.append(df.iloc[:, feature_indices].values)
                y.append(label_map[cls])
            except Exception as e:
                print(f"⚠️ ERROR in {file_path}: {e}")
                continue
    X = np.stack(X)
    y = np.array(y)
    return X, y

def save_results(results, model_name, window, feat_name, base_dir):
    out_dir = os.path.join(base_dir, "results_dl")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{window}_{feat_name}_{model_name}"
    np.save(os.path.join(out_dir, f"{prefix}_y_true.npy"), results['y_true'])
    np.save(os.path.join(out_dir, f"{prefix}_y_pred.npy"), results['y_pred'])
    np.save(os.path.join(out_dir, f"{prefix}_y_prob.npy"), results['y_prob'])

    # Confusion matrix
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AGGRESSIVE", "NORMAL", "CALM"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()

    # ROC
    y_true = results['y_true']
    y_prob = results['y_prob']
    n_classes = y_prob.shape[1]
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {prefix}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"))
    plt.close()

def plot_metrics(results, model_name, window, feat_name, base_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    out_dir = os.path.join(base_dir, "results_dl")
    prefix = f"{window}_{feat_name}_{model_name}"

    def safe_mean_std(data_list):
        max_len = max(len(x) for x in data_list)
        data_padded = np.full((len(data_list), max_len), np.nan)
        for i, lst in enumerate(data_list):
            data_padded[i, :len(lst)] = lst
        mean = np.nanmean(data_padded, axis=0)
        std = np.nanstd(data_padded, axis=0)
        return mean, std

    # Fold loss and acc
    all_train_loss = results['train_loss']
    all_val_loss = results['val_loss']
    all_train_acc = results['train_acc']
    all_val_acc = results['val_acc']

    for metric_name, train_list, val_list, ylabel in [
        ('loss', all_train_loss, all_val_loss, 'Loss'),
        ('acc', all_train_acc, all_val_acc, 'Accuracy')
    ]:
        train_mean, train_std = safe_mean_std(train_list)
        val_mean, val_std = safe_mean_std(val_list)

        epochs = np.arange(len(train_mean))
        plt.figure()
        plt.plot(epochs, train_mean, label="Train")
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
        plt.plot(epochs, val_mean, label="Validation")
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{prefix} {ylabel} (Mean ± Std)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_{metric_name}.png"))
        plt.close()
