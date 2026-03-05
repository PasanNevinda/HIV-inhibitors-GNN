import torch
import torch_geometric
import numpy as np


from src.models.model import GNN_GAT
from src.dataset.dataset_InMem import MoleculeInMemoryDataset

from tqdm import tqdm
import mlflow.pytorch
import time

from src.utils.utils import *
from configs.config import GNN_GAT_PARAMS


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = "/content/drive/MyDrive/GNNs/HIV inhibitors-GNN/data"
seed = 42
torch.manual_seed(seed)
torch_geometric.seed_everything(seed)


def train_one_epoch(model, dataloader, optimizer, loss_fun, epoch, should_log_cm=False):
    model.train()
    all_probs =[]
    all_label= []
    all_loss = 0.0
    for batch in tqdm(dataloader, desc="Training......"):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
        loss = loss_fun(pred, batch.y.float())
        loss.backward()
        optimizer.step()

        all_loss += loss.item() * batch.num_graphs
        all_probs.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_label.append(batch.y.cpu().detach().numpy())
    
    all_probs = np.concatenate(all_probs).ravel()
    all_label = np.concatenate(all_label).ravel()


    roc_auc = cal_matrics(all_label, all_probs, epoch, type="train")
    if(should_log_cm):
        log_confusion_matrix_and_roc(all_label, all_probs, prefix="Train")

    return all_loss / len(dataloader.dataset), roc_auc


def test(model, dataloader, loss_fun, epoch=None, mode="Validation", should_log_cm=False):
    model.eval()
    all_probs =[]
    all_label= []
    all_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{mode}......"):
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
            loss = loss_fun(pred, batch.y.float())

            all_loss += loss.item() * batch.num_graphs
            all_probs.append(torch.sigmoid(pred).cpu().detach().numpy())
            all_label.append(batch.y.cpu().detach().numpy())
    
    all_probs = np.concatenate(all_probs).ravel()
    all_label = np.concatenate(all_label).ravel()


    roc_auc = cal_matrics(all_label, all_probs, epoch, type=mode)
    if(should_log_cm):
        log_confusion_matrix_and_roc(all_label, all_probs, prefix=mode)

    return all_loss / len(dataloader.dataset) , roc_auc


def run_one_training(params, epochs=100):
    
    print("loading datasets.........")
    train_dataset = MoleculeInMemoryDataset(root=ROOT, filename="HIV_train_val.csv")
    test_dataset = MoleculeInMemoryDataset(root=ROOT, filename="HIV_test.csv", test=True)
    
    #derived
    node_feature_size = train_dataset.num_node_features
    params["model_edge_dim"] = train_dataset.num_edge_features

    
    train_loader, val_loader, test_loader, pos_weight = get_train_val_test_loaders_posweight(train_dataset, test_dataset, val_portion=0.12, batch_size=params["batch_size"], seed=seed)
   
    model_params = {k:v for k,v in params.items() if k.startswith("model_")}
    model = GNN_GAT(feature_size=node_feature_size, parameters=model_params)
    model = model.to(device)

    pos_weight = pos_weight.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=params["sgd_learning_rate"],
        momentum=params["sgd_momentem"],
        weight_decay=params["sgd_weight_dacay"]
    )
    
    start_epoch, best_roc_auc, saved_run_id = load_checkpoint(model, optimizer, device)
    
    if saved_run_id is not None:
        print("Resuming MLflow run:", saved_run_id)
        mlflow.start_run(run_id=saved_run_id)
        run_id = saved_run_id
    else:
        mlflow.end_run()  
        run = mlflow.start_run(run_name="GAT_GNN")
        run_id = run.info.run_id
        print("Starting new MLflow run:", run_id)

        for key in params.keys():
            mlflow.log_param(key=key, value=params[key])
    
        mlflow.log_param(key="weighted sampler", value=True)
        mlflow.log_param("pos_weight", float(pos_weight.item()))

    for epoch in range(start_epoch, epochs):
        loss, train_roc_auc = train_one_epoch(model,train_loader, optimizer, loss_fn, epoch)
        print(f"Epoch: {epoch} | Train loss: {loss}")
        mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

        if((epoch + 1) % 5 == 0):
            val_loss, val_roc_auc = test(model, val_loader, loss_fn, epoch)
            save_checkpoint(model, optimizer, epoch, val_roc_auc, run_id)
            print(f"Epoch: {epoch} | Validation loss: {val_loss}")
            mlflow.log_metric(key="Validation loss", value=float(val_loss), step=epoch)

            if val_roc_auc > best_roc_auc:
                best_roc_auc = val_roc_auc
                save_checkpoint(model, optimizer, epoch, best_roc_auc,  run_id, "best_model.tar")
                mlflow.log_metric("best_val_roc_auc", best_roc_auc)
                print(f"New best AUC: {val_roc_auc:.4f} at epoch {epoch}")

        
    _, _, _ = load_checkpoint(model, optimizer, device,"best_model.tar")

    train_loss, _ = test(model, train_loader, loss_fn,1000, mode="Train",should_log_cm=True)
    validation_loss, _ = test(model, val_loader, loss_fn,1000, mode="Validation",should_log_cm=True)
    test_loss, _ = test(model, test_loader, loss_fn,1000, mode="Test",should_log_cm=True)
    mlflow.log_metric(key="Train Final loss", value=float(train_loss), step=1000)
    mlflow.log_metric(key="Validation Final loss", value=float(validation_loss), step=1000)
    mlflow.log_metric(key="Test Final loss", value=float(test_loss), step=1000)
    

if __name__ == "__main__":
    run_one_training(GNN_GAT_PARAMS, epochs=100)