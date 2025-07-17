import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import os


def train_and_evaluate(X, y, model_fn, input_dim, num_classes, folds, batch_size, epochs,
                       model_name, dataset_name, device, results_dir):

    all_train_loss, all_val_loss = [], []
    all_train_acc, all_val_acc = [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"  Fold {fold+1}")
        model = model_fn(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_val_loss = float('inf')
        patience, wait = 10, 0

        train_loss_hist, val_loss_hist = [], []
        train_acc_hist, val_acc_hist = [], []

        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
        y_train = torch.tensor(y[train_idx], dtype=torch.long).to(device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        y_val = torch.tensor(y[val_idx], dtype=torch.long).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)

            # Accuracy
            train_preds = torch.argmax(outputs, dim=1)
            val_preds = torch.argmax(val_outputs, dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            val_acc = (val_preds == y_val).float().mean().item()

            train_loss_hist.append(loss.item())
            val_loss_hist.append(val_loss.item())
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)

            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation
        y_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
        y_prob = torch.softmax(val_outputs, dim=1).cpu().numpy()
        y_true = y_val.cpu().numpy()

        all_train_loss.append(train_loss_hist)
        all_val_loss.append(val_loss_hist)
        all_train_acc.append(train_acc_hist)
        all_val_acc.append(val_acc_hist)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        
                # Fold metriklerini hesapla
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, average='macro')
        prec = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # Fold metriklerini diske kaydet
        fold_metrics_dir = os.path.join(results_dir, 'fold_metrics')
        os.makedirs(fold_metrics_dir, exist_ok=True)
        prefix = f"{dataset_name}_{model_name}"
        np.save(os.path.join(fold_metrics_dir, f"{prefix}_fold{fold}_accuracy.npy"), np.array(acc))
        np.save(os.path.join(fold_metrics_dir, f"{prefix}_fold{fold}_recall.npy"), np.array(rec))
        np.save(os.path.join(fold_metrics_dir, f"{prefix}_fold{fold}_precision.npy"), np.array(prec))
        np.save(os.path.join(fold_metrics_dir, f"{prefix}_fold{fold}_f1.npy"), np.array(f1))


    return {
        'train_loss': all_train_loss,
        'val_loss': all_val_loss,
        'train_acc': all_train_acc,
        'val_acc': all_val_acc,
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_prob': np.array(all_y_prob)
    }
