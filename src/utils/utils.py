import os
import torch
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import mlflow


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, \
                            precision_score, recall_score, roc_auc_score, roc_curve, \
                            auc, classification_report

BASE_DIR = '/content/drive/MyDrive/GNNs/HIV inhibitors-GNN'
CHECKPOINT_DIR = f"{BASE_DIR}/outputs/checkpoints"

def save_checkpoint(model, optimizer, epoch, best_roc_auc, run_id, filename="latest_checkpoint.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_roc_auc': best_roc_auc,
        'run_id': run_id
    }
    path = f"{CHECKPOINT_DIR}/{filename}"
    torch.save(checkpoint, path)
    print(f"checkpoint saved: {path}")
    return path

def load_checkpoint(model, optimizer, device, filename="latest_checkpoint.tar"):
    path = f"{CHECKPOINT_DIR}/{filename}"
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"resumed from epoch {checkpoint['epoch']} | best ROC-AUC: {checkpoint['best_roc_auc']:.4f}")
        return checkpoint['epoch'] + 1, checkpoint['best_roc_auc'], checkpoint.get('run_id', None)
    else:
        print("no checkpoint found. starting from epoch 0.")
        return 0, 0.0, None 

def cal_matrics(label, probs, epoch, type):
    pred = (probs > 0.5).astype(int)
    
    acc       = accuracy_score(label, pred)
    precision = precision_score(label, pred, pos_label=1, zero_division=0)
    recall    = recall_score(label, pred, pos_label=1, zero_division=0)
    f1        = f1_score(label, pred, pos_label=1, zero_division=0)

    try:
        roc_auc = roc_auc_score(label, probs)
    except ValueError:
        roc_auc = 0.0

    print(f"Type:{type} | Epoch:{epoch} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

    mlflow.log_metric(key=f"Precision-{type}", value=precision, step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=recall, step=epoch)
    mlflow.log_metric(key=f"F1-{type}", value=f1, step=epoch)
    mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc_auc), step=epoch)
    
    return roc_auc


def log_confusion_matrix_and_roc(label, prob, epoch=None, prefix="val"):
    pred = (prob > 0.5).astype(int)
   
    cm = confusion_matrix(label, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix ({prefix})')
    
    if epoch is not None:
        mlflow.log_figure(fig, f"confusion_matrix/{prefix}_epoch_{epoch}.png")
    else:
        mlflow.log_figure(fig, f"confusion_matrix/{prefix}_final.png")
    plt.close(fig)

    
    fpr, tpr, _ = roc_curve(label, prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve ({prefix})')
    ax.legend()
    
    if epoch is not None:
        mlflow.log_figure(fig, f"roc_curve/{prefix}_epoch_{epoch}.png")
    else:
        mlflow.log_figure(fig, f"roc_curve/{prefix}_final.png")
    plt.close(fig)

   
    report = classification_report(label, pred, output_dict=True)
    mlflow.log_dict(report, f"classification_report/{prefix}.json")


def get_train_val_test_loaders_posweight(train_dataset, test_dataset, val_portion, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(train_dataset) * val_portion)

    train_subset, val_subset = random_split(
        train_dataset, 
        [len(train_dataset) - val_size, val_size],
        generator=generator
    )
    print(f"train_set_size:{len(train_subset)} | val_set_size:{len(val_subset)}")
    labels = np.array([d.y.item() for d in train_subset])
    class_counts = np.bincount(labels)

    pos_weight = torch.tensor([class_counts[0]/ class_counts[1]])

    print(f"Class distribution: {class_counts[0]} negatives, {class_counts[1]} positives")
    print(f"Recommended pos_weight = {pos_weight:.1f}")

    sampler = WeightedRandomSampler(
        weights=1.0 / class_counts[labels],
        num_samples=len(labels),
        replacement=True,
        generator=generator             
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False, 
        
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader, pos_weight