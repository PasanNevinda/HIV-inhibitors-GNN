import os
import torch
from torch_geometric.data import InMemoryDataset
import pandas as pd
import numpy as np
from rdkit import Chem
import deepchem as dc

from tqdm import tqdm

class MoleculeInMemoryDataset_DC(InMemoryDataset):
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
        return f'{prefix}_molecules_deepchem.pt'
    
    def download(self):
        pass

    def process(self):
        raw_path = self.raw_paths[0]
        df = pd.read_csv(raw_path)
        
        featurizer = dc.feat.MolGraphConvFeaturizer(
                use_edges=True,         
                use_chirality=False,
                use_partial_charge=False
            )

        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is not None:
                g = featurizer.featurize(mol)[0]
                graph = g.to_pyg_graph()
                graph.y = torch.tensor([row["HIV_active"]], dtype=torch.float32)
                data_list.append(graph)

       
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f"Saving {len(data_list)} graphs to {self.processed_paths[0]}")
        self.save(data_list, self.processed_paths[0])




