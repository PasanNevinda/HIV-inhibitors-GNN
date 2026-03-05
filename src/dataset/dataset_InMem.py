import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

class MoleculeInMemoryDataset(InMemoryDataset):
    def __init__(self, root, filename, test=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.filename = filename
        self.test = test
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])   

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        prefix = "test" if self.test else "train_val"
        return f'{prefix}_molecules.pt'
    
    def download(self):
        pass

    def process(self):
        raw_path = self.raw_paths[0]
        df = pd.read_csv(raw_path)
        
        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue  # skip invalid SMILES

            x     = self._get_node_features(mol)
            edge_index = self._get_adjacency_info(mol)
            edge_attr  = self._get_edge_features(mol)
            y     = self._get_labels(row["HIV_active"])

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                smiles=row["smiles"]
            )
            data_list.append(data)

       
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f"Saving {len(data_list)} graphs to {self.processed_paths[0]}")
        self.save(data_list, self.processed_paths[0])

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


