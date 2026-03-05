import os
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import pandas as pd
from rdkit.Chem import rdmolops
from rdkit import Chem
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
        data = pd.read_csv(self.raw_paths[0])

        if self.test:
          return [f"data_test_{i}.pt" for i in range(len(data))]
        
        return [f"data_{i}.pt" for i in range(len(data))]

    def download(self):
        pass

    def process(self):
        raw_path = self.raw_paths[0]
        print("row path {}".format(raw_path))
        
        self.data = pd.read_csv(raw_path)
        
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol["smiles"])

            node_feats   = self._get_node_features(mol_obj)
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
            

    def _get_node_features(self, mol_obj: Chem.Mol):
        '''
            return matrix of shape [num_nodes, num_node_attributes]
        '''
        node_features = []
        for atom in mol_obj.GetAtoms():
            node_f = []
            # atomic num
            node_f.append(atom.GetAtomicNum())
            # degree
            node_f.append(atom.GetDegree())
            # total degree
            node_f.append(atom.GetTotalDegree())
            # Total number of hydrogens
            node_f.append(atom.GetTotalNumHs())
            # formal charge
            node_f.append(atom.GetFormalCharge())
            # hibridization
            node_f.append(atom.GetHybridization())
            # Number of radical electrons
            node_f.append(atom.GetNumRadicalElectrons())
            # Aromaticity
            node_f.append(atom.GetIsAromatic())
            # mass 
            node_f.append(atom.GetMass())
            # in ring
            node_f.append(atom.IsInRing())
            # Chirality
            node_f.append(atom.GetChiralTag())
        
            node_features.append(node_f)
        
        node_features = np.asarray(node_features)

        return torch.tensor(node_features, dtype=torch.float32) 

    def _get_edge_features(self, mol_obj: Chem.Mol):
        ''' returns matrix of shape [num_edges, num_edge_features] '''
        edges_attributes = []
        for bond in mol_obj.GetBonds():
            edge_attr = [
                bond.GetBondTypeAsDouble(),
                bond.GetIsAromatic(),
                bond.IsInRing(),
                bond.GetIsConjugated(),
                int(bond.GetStereo())
            ]

            edges_attributes += [edge_attr, edge_attr]
        
        edges_attributes = np.asarray(edges_attributes)
        return torch.tensor(edges_attributes, dtype=torch.float32)

    def _get_adjacency_info(self, mol_obj: Chem.Mol):
        ''' return matrix of shape [2, num_edges]'''
        src, dest = [], []
        for bond in mol_obj.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            src += [i, j]
            dest += [j, i]
        return torch.tensor([src, dest], dtype=torch.long)

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        path = os.path.join(self.processed_dir,
        f"data_test_{idx}.pt" if self.test else f"data_{idx}.pt" )
        return torch.load(path, weights_only=False)

'''
  dataset = MoleculeDataset(root="/content/data", filename="HIV.csv")
    this create
      self.root = "/content/data"
      self.raw_dir = "/content/data/raw"
      self.processed_dir = "/content/data/processed"
''' 


