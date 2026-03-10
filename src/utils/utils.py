import os
import torch
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import mlflow


from sklearn.metrics import precision_score, confusion_matrix,  \
                            recall_score, roc_auc_score, roc_curve, \
                            auc, classification_report, average_precision_score, precision_recall_curve

BASE_DIR = '/content/drive/MyDrive/GNNs/HIV inhibitors-GNN'
CHECKPOINT_DIR = f"{BASE_DIR}/outputs/checkpoints"

def save_checkpoint(model, optimizer, epoch, average_precision, loss, run_id, filename="latest_checkpoint.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'average_precision': average_precision,
        'loss': loss,
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
        
        print(f"resumed from epoch {checkpoint['epoch']} | Average Precision: {float(checkpoint['average_precision']):.4f} | Loss: {float(checkpoint['loss']):.4f}")
        return checkpoint['epoch'] + 1, checkpoint['average_precision'], checkpoint['loss'], checkpoint.get('run_id', None)
    else:
        print("no checkpoint found. starting from epoch 0.")
        return 0, -float('inf'), float('inf'), None 

def cal_matrics(label, probs, epoch, type, Final=False):
    thresholds = [0.1, 0.2, 0.3, 0.5]
    recall_at_t = {}
    precision_at_t = {}
    for t in thresholds:
        pred = (probs > t).astype(int)
        recall_at_t[t]    = recall_score(label, pred, pos_label=1, zero_division=0)
        precision_at_t[t] = precision_score(label, pred, pos_label=1, zero_division=0)
    try:
        roc_auc = roc_auc_score(label, probs)
    except ValueError:
        roc_auc = 0.0

    average_precision = average_precision_score(label, probs)
    
    if Final:
        print(f"Final Results =>  Type:{type}| AP: {average_precision:.4f} | AUC: {roc_auc:.4f}", end=" ")
        for t in thresholds:
            print(f"| Final Recall@{t}: {recall_at_t[t]} | Final Precision@{t}: {precision_at_t[t]}", end=" ")
        print("\n")

        mlflow.log_metric(key=f"Final Average-Precision-{type}", value=average_precision)
        mlflow.log_metric(key=f"Final ROC-AUC-{type}", value=float(roc_auc))
        for t in thresholds:
            mlflow.log_metric(f"Final Recall_at_{t}-{type}", recall_at_t[t])
            mlflow.log_metric(f"Final Precision_at_{t}-{type}", precision_at_t[t])
        return average_precision

    print(f"Type:{type}| Epoch:{epoch} | AP: {average_precision:.4f} | AUC: {roc_auc:.4f}", end=" ")
    for t in thresholds:
        print(f"| Recall@{t}: {recall_at_t[t]} | Precision@{t}: {precision_at_t[t]}", end=" ")
    print("\n")

    mlflow.log_metric(key=f"Average-Precision-{type}", value=average_precision, step=epoch)
    mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc_auc), step=epoch)
    for t in thresholds:
            mlflow.log_metric(f"Recall_at_{t}-{type}", recall_at_t[t], step=epoch)
            mlflow.log_metric(f"Precision_at_{t}-{type}", precision_at_t[t], step=epoch)
    
    return average_precision


def log_confusion_matrix_and_pr_curve(label, prob, prefix="val"):
    thresholds = [0.1, 0.2, 0.3, 0.5]
    for t in thresholds:
        pred = (prob > t).astype(int)
    
        # confusion matrix
        cm = confusion_matrix(label, pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix ({prefix} - threshold at {t})')

        mlflow.log_figure(fig, f"confusion_matrix/{prefix}_threshold_at_{t}_final.png")
        plt.close(fig)
    
        # classification report
        report = classification_report(label, pred, output_dict=True)
        mlflow.log_dict(report, f"classification_report/{prefix}_threshold_at_{t}.json")

    # roc curve
    fpr, tpr, _ = roc_curve(label, prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve ({prefix})')
    ax.legend()

    mlflow.log_figure(fig, f"roc_curve/{prefix}_final.png")
    plt.close(fig)

    ap = average_precision_score(label, prob)
    
    # average precision curve
    precision, recall, _ = precision_recall_curve(label, prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP = {ap:.2f})')
    
    # Add no-skill baseline (random classifier: precision = fraction of positives)
    no_skill = np.sum(label) / len(label)
    ax.plot([0, 1], [no_skill, no_skill], color='gray', linestyle='--', label='No Skill')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve ({prefix})')
    ax.legend(loc='lower left')
    ax.grid(True)
    
    mlflow.log_figure(fig, f"pr_curve/{prefix}_final.png")
    plt.close(fig)


def get_train_val_test_loaders_posweight(train_dataset, test_dataset, val_portion, batch_size, seed, use_pos_weight, use_weighted_sampler):
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(train_dataset) * val_portion)

    train_subset, val_subset = random_split(
        train_dataset, 
        [len(train_dataset) - val_size, val_size],
        generator=generator
    )
    print(f"train_set_size:{len(train_subset)} | val_set_size:{len(val_subset)}")
    labels = np.array([int(d.y.item()) for d in train_subset])
    class_counts = np.bincount(labels)
    
    print(f"Class Counts: {class_counts}")

    pos_weight = torch.sqrt(torch.tensor([class_counts[0] / class_counts[1]])) if use_pos_weight else torch.tensor([1.0])

    print(f"Class distribution: {class_counts[0]} negatives(0), {class_counts[1]} positives(1)")
    print(f"pos_weight = {pos_weight.item():.1f}")

    sampler = WeightedRandomSampler(
        weights=1.0 / class_counts[labels],
        num_samples=len(labels),
        replacement=True,
        generator=generator             
    )

    if use_weighted_sampler:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False, 
            
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True, 
            
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