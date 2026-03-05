import os
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import pandas as pd
from rdkit.Chem import rdmolops
from rdkit import Chem
import deepchem as dc
from tqdm import tqdm



class MoleculeDataSet(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, pre_filter=None):

        self.filename=filename
        self.test=test
        super().__init__(root, transform, pre_transform, pre_filter)

    '''
        root is where dataset is stored (processed dir)
    '''
    @property
    def raw_file_names(self):
      "If this file exists in raw_dir, the download is not occured."
      return self.filename

    @property
    def processed_file_names(self):
        ''' if files are found in this process method skiped'''
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        raw_path = self.raw_paths[0]
        print("row path {}".format(raw_path))
        
        self.data = pd.read_csv(raw_path).reset_index()
        self.featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            smiles = row["smiles"]
            label = row["HIV_active"] 

            f = self.featurizer.feturize([smiles])[0]
            node_feats   = torch.tensor(f.)
            edge_feats   = self._get_edge_features(mol_obj)
            edge_index   = self._get_adjacency_info(mol_obj)
            label        = self._get_labels(mol["HIV_active"])

            # Create a PyTorch Geometric Data object
            data = Data(
                x         = node_feats,          # node features (shape: [num_nodes, num_node_features])
                edge_index = edge_index,          # shape: [2, num_edges] 
                edge_attr  = edge_feats,          # edge features (shape: [num_edges, num_edge_features])
                y         = label,                # the target label (0 or 1 for HIV active/inactive)
                smiles    = mol["smiles"]         # keep original SMILES for debugging
            )

           
           # Save this graph as a .pt file
            if self.test:
              torch.save(data, os.path.join(
                    self.processed_dir,
                    f'data_test_{index}.pt'))
            else:
              torch.save(data, os.path.join(
                    self.processed_dir,
                    f'data_{index}.pt'))

            
    def _get_labels(self, label):
      label = np.asarray([label])
      return torch.tensor(label, dtype=torch.int64)
   

    def len(self):
        return self.length 

    def get(self, idx):
        path = os.path.join(self.processed_dir,
        f"data_test{idx}" if self.test else f"data_{idx}" )
        return torch.load(path, weights_only=False)

'''
  dataset = MoleculeDataset(root="/content/data", filename="HIV.csv")
    this create
      self.root = "/content/data"
      self.raw_dir = "/content/data/raw"
      self.processed_dir = "/content/data/processed"
''' 


