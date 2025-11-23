import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset#, DataLoader
from torch_geometric.loader import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
from openbabel import pybel                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def canonical(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels
def generate_3d(mol):
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        
        result = AllChem.EmbedMolecule(mol3d, params)

        if result != 0:
            return None, False 

        #AllChem.UFFOptimizeMolecule(mol3d)
        pos = mol3d.GetConformer().GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
        pos = pos - pos.mean(dim=0, keepdim=True)
        return pos,  True

    except Exception as e:
        raise e
        return None, False

def generate_3d_openbabel(smiles):
    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D()
        coords = torch.tensor([list(a.coords) for a in mol.atoms], dtype=torch.float)
        return coords, True
    except Exception:
        
        return None, False

class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task, use3D=False):
        super(Dataset, self).__init__()
        #self.smiles_data, self.labels = read_smiles(data_path, target, task)
        
        
        self.use3D= use3D
        smiles_list, labels = read_smiles(data_path, target, task)
        filtered_smiles, filtered_labels = [], []
        invalid_count = 0
        
        for s,l in zip(smiles_list,labels):
            try:
                mol = Chem.MolFromSmiles(s)
                mol = Chem.AddHs(mol)
            except:
                invalid_smiles += 1
                continue
            
            if use3D:
                pos, flag = generate_3d(mol)
            
                if not flag or pos is None:
                    invalid_count += 1
                    continue
            
                filtered_smiles.append((s, pos))
                filtered_labels.append(l)
                                
            else:
                filtered_smiles.append(s)
                filtered_labels.append(l)
        
        self.smiles_data = filtered_smiles
        self.labels = filtered_labels
        print(f" Filtered out {invalid_count} invalid SMILES.")
        
        
        
        self.task = task

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        if self.use3D:
            s, pos =  self.smiles_data[index]
        else:
            s =  self.smiles_data[index]
        
        mol = Chem.MolFromSmiles(s)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        if self.use3D:
            if pos is None:
                raise RuntimeError(f"pos became None at index {index}, SMILES={s}")
            data.pos = pos
        
        
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting, use_3D= False
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.use_3D= use_3D
        assert splitting in ['random', 'scaffold', 'tox21']
    
    def _load_tox21_split(self, dataset):
        """
        Read train_smiles.txt, valid_smiles.txt, test_smiles.txt
        and map each SMILES to index in dataset.smiles_data
        """
        def read_txt(path):
            with open(path) as f:
                return [l.strip() for l in f if l.strip()]

        root = self.data_path
        directory = os.path.dirname(root)
        
        train_smiles = read_txt(os.path.join(directory, "train_smiles.txt"))
        valid_smiles = read_txt(os.path.join(directory, "valid_smiles.txt"))
        test_smiles  = read_txt(os.path.join(directory, "test_smiles.txt"))

        # Map SMILES ? list index in dataset
        if self.use_3D:
            smiles_to_index = {canonical(s): i for i, (s,_) in enumerate(dataset.smiles_data)}
        else:
            smiles_to_index = {canonical(s): i for i, s in enumerate(dataset.smiles_data)}

        train_idx = [smiles_to_index[s] for s in train_smiles if s in smiles_to_index]
        valid_idx = [smiles_to_index[s] for s in valid_smiles if s in smiles_to_index]
        test_idx  = [smiles_to_index[s] for s in test_smiles  if s in smiles_to_index]

        print(f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

        return train_idx, valid_idx, test_idx

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task, use3D=self.use_3D)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)
        
        elif self.splitting == 'tox21':
            train_idx, valid_idx, test_idx = self._load_tox21_split(train_dataset)


        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
