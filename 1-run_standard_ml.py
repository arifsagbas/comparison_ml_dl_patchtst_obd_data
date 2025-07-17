import os
import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# Tüm tohumları sabitle
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

# Özellik grupları (indisler 0'dan başlıyor)
FEATURE_GROUPS = {
    'ECU': list(range(24, 46)),
    'ECU+Motion': list(range(0, 24)) + list(range(24, 46)),
    'ECU+Motion+Overpass': list(range(0, 24)) + list(range(24, 46)) + list(range(46, 50)) + list(range(63, 66)),
    'All': list(range(0, 24)) + list(range(24, 46)) + list(range(46, 50)) + list(range(63, 66)) + list(range(50, 63))
}

# Modeller
MODELS = {
    'LR': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'NB': GaussianNB(),
    'ANN': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
}

# Dosya yolları
dataset_paths = [
    "C:\\Users\\Ensar\\Desktop\\OBD\\dataset3s65f_normalized.csv",
    "C:\\Users\\Ensar\\Desktop\\OBD\\dataset5s65f_normalized.csv",
    "C:\\Users\\Ensar\\Desktop\\OBD\\dataset7s65f_normalized.csv",
    "C:\\Users\\Ensar\\Desktop\\OBD\\dataset9s65f_normalized.csv"
]

results_dir = "C:\\Users\\Ensar\\Desktop\\OBD\\results_ml"
os.makedirs(results_dir, exist_ok=True)

all_reports = []

for dataset_path in dataset_paths:
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    df = pd.read_csv(dataset_path)
    X_all = df.drop(columns=[df.columns[-1]]).values
    y_all = LabelEncoder().fit_transform(df.iloc[:, -1].values)

    for group_name, feature_indices in FEATURE_GROUPS.items():
        X = X_all[:, feature_indices]

        for model_name, model in MODELS.items():
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            fold_metrics = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
            all_y_true = []
            all_y_pred = []
            all_y_prob = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_all)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_all[train_idx], y_all[val_idx]

                clf = model
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                y_prob = clf.predict_proba(X_val)

                # Skorlar
                acc = accuracy_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred, average='macro')
                precision = precision_score(y_val, y_pred, average='macro')
                f1 = f1_score(y_val, y_pred, average='macro')

                fold_metrics['accuracy'].append(acc)
                fold_metrics['recall'].append(recall)
                fold_metrics['precision'].append(precision)
                fold_metrics['f1'].append(f1)

                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred)
                all_y_prob.extend(y_prob)

                # Fold verilerini kaydet
                base_fold = f"{dataset_name}_{group_name}_{model_name}_fold{fold_idx}"
                np.save(os.path.join(results_dir, f"{base_fold}_y_true.npy"), y_val)
                np.save(os.path.join(results_dir, f"{base_fold}_y_pred.npy"), y_pred)
                np.save(os.path.join(results_dir, f"{base_fold}_y_prob.npy"), y_prob)

            # Fold'ların toplam verilerini de kaydet
            base_name = f"{dataset_name}_{group_name}_{model_name}"
            np.save(os.path.join(results_dir, f"{base_name}_y_true.npy"), np.array(all_y_true))
            np.save(os.path.join(results_dir, f"{base_name}_y_pred.npy"), np.array(all_y_pred))
            np.save(os.path.join(results_dir, f"{base_name}_y_prob.npy"), np.array(all_y_prob))
            
            # Fold metriklerini kaydet
            for metric_name, scores in fold_metrics.items():
                metric_array = np.array(scores)
                np.save(os.path.join(results_dir, f"{base_name}_{metric_name}.npy"), metric_array)


            # ROC eğrisi (Tüm veri)
            y_true_arr = np.array(all_y_true)
            y_prob_arr = np.array(all_y_prob)
            n_classes = y_prob_arr.shape[1]

            plt.figure()
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve((y_true_arr == i).astype(int), y_prob_arr[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {base_name}")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{base_name}_roc.png"))
            plt.close()

            # Ortalama skorlar
            summary = {
                'dataset': dataset_name,
                'features': group_name,
                'model': model_name,
                'accuracy': np.mean(fold_metrics['accuracy']),
                'recall': np.mean(fold_metrics['recall']),
                'precision': np.mean(fold_metrics['precision']),
                'f1': np.mean(fold_metrics['f1'])
            }
            all_reports.append(summary)

# Toplu özet rapor
report_df = pd.DataFrame(all_reports)
report_df.to_csv(os.path.join(results_dir, "summary_report.csv"), index=False)
