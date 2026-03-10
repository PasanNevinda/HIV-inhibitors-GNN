import torch
import torch_geometric
import numpy as np


from src.models.model import GNN_GAT
from src.dataset.dataset_InMem_DeepChem import MoleculeInMemoryDataset_DC

from tqdm import tqdm
import mlflow.pytorch
import time

from src.utils.utils import *
from configs.config import GNN_GAT_PARAMS

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = "/content/drive/MyDrive/GNNs/HIV inhibitors-GNN/data" 
seed = 42
torch.manual_seed(seed)
torch_geometric.seed_everything(seed)




def parse_args():
    parser = argparse.ArgumentParser(description="GAT GNN Training Script")
    parser.add_argument('--use_pos_weight', type=int, default=1, 
                        help="Whether to use pos_weight in BCEWithLogitsLoss (1=yes, 0=no)")
    parser.add_argument('--use_weighted_sampler', type=int, default=1, 
                        help="Whether to use WeightedRandomSampler for training loader (1=yes, 0=no)")
    
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Number of epochs")

    return parser.parse_args()



def train_one_epoch(model, dataloader, optimizer, loss_fun, epoch):
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


    average_precision = cal_matrics(all_label, all_probs, epoch, type="train")

    return all_loss / len(dataloader.dataset), average_precision


def test(model, dataloader, loss_fun, epoch=None, mode="Validation", should_log_cm=False, Final=False):
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


    average_precision = cal_matrics(all_label, all_probs, epoch, type=mode, Final=Final)
    if(should_log_cm):
        log_confusion_matrix_and_pr_curve(all_label, all_probs, prefix=mode)

    return all_loss / len(dataloader.dataset) , average_precision


def run_one_training(params, epochs=100, use_pos_weight=False, use_weighted_sampler=False):
    
    print("loading datasets.........")
    train_dataset = MoleculeInMemoryDataset_DC(root=ROOT, filename="HIV_train_val.csv")
    test_dataset = MoleculeInMemoryDataset_DC(root=ROOT, filename="HIV_test.csv", test=True)
    
    #derived
    print("\nDerived parameters......")
    node_feature_size = train_dataset.num_node_features
    params["model_edge_dim"] = train_dataset.num_edge_features

    print("\nGet loaders..................")
    train_loader, val_loader, test_loader, pos_weight = get_train_val_test_loaders_posweight(train_dataset, test_dataset, val_portion=0.12, batch_size=params["batch_size"], seed=seed, use_pos_weight=use_pos_weight, use_weighted_sampler=use_weighted_sampler)
   
    print("\nCreate Model.................")
    model_params = {k:v for k,v in params.items() if k.startswith("model_")}
    model = GNN_GAT(feature_size=node_feature_size, parameters=model_params)
    model = model.to(device)

    pos_weight = pos_weight.to(device)

    print("\nCreate loss function and Optimizer.................")
    
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=params["sgd_learning_rate"],
        momentum=params["sgd_momentem"],
        weight_decay=params["sgd_weight_dacay"]
    )
    
    print("\nLoad Checkpoint.................")
    run_name = f"GAT-GNN- With_WeightedSampler{use_weighted_sampler} -WithPosWeigh{use_pos_weight} -DeepChem-features"
    start_epoch, best_average_precision, best_loss, saved_run_id = load_checkpoint(model, optimizer, device, filename=f"{run_name}-latest_checkpoint.tar")
    
    if saved_run_id is not None:
        print("Resuming MLflow run:", saved_run_id)
        mlflow.start_run(run_id=saved_run_id)
        run_id = saved_run_id
    else:
        mlflow.end_run()  
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        print("Starting new MLflow run:", run_id)

        print("\nLog Params in MLflow.........................")
        for key in params.keys():
            mlflow.log_param(key=key, value=params[key])
    
        mlflow.log_param("Use weighted sampler", use_weighted_sampler)
        mlflow.log_param("Use Pos weight", use_pos_weight)
        mlflow.log_param("pos_weight", float(pos_weight.item()))

        mlflow.log_param("Features", "DeepChem Features")

    print("\n\n\nStart training loop.....................\n\n")
    for epoch in range(start_epoch, epochs):
        loss, train_average_precision = train_one_epoch(model,train_loader, optimizer, loss_fn, epoch)
        print(f"Epoch: {epoch} | Train loss: {loss}\n\n")
        mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

        if((epoch + 1) % 5 == 0):
            print(f"-----------------------------Validation Start at {epoch}-------------------------------")
            val_loss, val_average_precision = test(model, val_loader, loss_fn, epoch)
            print("Saving Checkpoint..........................")
            save_checkpoint(model, optimizer, epoch, best_average_precision, best_loss, run_id, filename=f"{run_name}-latest_checkpoint.tar")
            print(f"Epoch: {epoch} | Validation loss: {val_loss}\n")
            mlflow.log_metric(key="Validation loss", value=float(val_loss), step=epoch)

            if val_average_precision > best_average_precision:
                best_average_precision = val_average_precision
                print(f"################## Saving Best model(using avg precision) so far at {epoch} ############################")
                save_checkpoint(model, optimizer, epoch, best_average_precision, val_loss, run_id, filename=f"{run_name}-best_model-ap.tar")
                mlflow.log_metric("best_average_precision", float(best_average_precision))
                print(f"New best Average Precision: {best_average_precision:.4f} at epoch {epoch}")
                print("################### End Saving Best model(ap) ###########################\n\n")
            
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"################## Saving Best model(using loss) so far at {epoch} ############################")
                save_checkpoint(model, optimizer, epoch, val_average_precision, best_loss, run_id, filename=f"{run_name}-best_model-loss.tar")
                mlflow.log_metric("best_val_loss", float(best_loss))
                print(f"New best loss: {best_loss:.4f} at epoch {epoch}")
                print("################### End Saving Best model(loss) ###########################\n\n")

            print("------------------------------End Validation-------------------------\n\n")

    print("\n\nEnd training loop.....................\n\n") 

    print("Start All Evaluations..........................................\n\n")
    print("Best Average Precision model evaluation")
    _, _, _, _ = load_checkpoint(model, optimizer, device, filename=f"{run_name}-best_model-ap.tar")

    train_loss, _ = test(model, train_loader, loss_fn, mode="Train-ap model",should_log_cm=True, Final=True)
    validation_loss, _ = test(model, val_loader, loss_fn, mode="Validation-ap model",should_log_cm=True, Final=True)
    test_loss, _ = test(model, test_loader, loss_fn, mode="Test-ap model",should_log_cm=True, Final=True)
    mlflow.log_metric(key="Train Final loss-ap model", value=float(train_loss))
    mlflow.log_metric(key="Validation Final loss-ap model", value=float(validation_loss))
    mlflow.log_metric(key="Test Final loss-ap model", value=float(test_loss))

    print("\n\nBest Loss model evaluation")
    _, _, _, _ = load_checkpoint(model, optimizer, device, filename=f"{run_name}-best_model-loss.tar")

    train_loss, _ = test(model, train_loader, loss_fn, mode="Train-loss model",should_log_cm=True, Final=True)
    validation_loss, _ = test(model, val_loader, loss_fn, mode="Validation-loss model",should_log_cm=True, Final=True)
    test_loss, _ = test(model, test_loader, loss_fn, mode="Test-loss model",should_log_cm=True, Final=True)
    mlflow.log_metric(key="Train Final loss- loss model", value=float(train_loss))
    mlflow.log_metric(key="Validation Final loss- loss model", value=float(validation_loss))
    mlflow.log_metric(key="Test Final loss- loss model", value=float(test_loss))
    

if __name__ == "__main__":
    args = parse_args()
    use_pos_weight = bool(args.use_pos_weight)
    use_weighted_sampler = bool(args.use_weighted_sampler)
    epochs = args.epochs

    print(f"Using pos_weight: {use_pos_weight}, Using weighted sampler: {use_weighted_sampler}")

    run_one_training(GNN_GAT_PARAMS, epochs=epochs, use_pos_weight=use_pos_weight, use_weighted_sampler=use_weighted_sampler)