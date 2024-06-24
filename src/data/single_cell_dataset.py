import os, json, copy, sys
from pathlib import Path
import ml_collections as mlc
import numpy as np
import pandas as pd
import logging

import lmdb, pickle, dgl, time
import torch
import torch.nn.functional as F
from typing import *
# import scvi

from unicore.data import UnicoreDataset, data_utils
import torch_geometric as pyg

from models.gene_tokenizer import GeneVocab

logger = logging.getLogger(__file__)

class ScDataset(UnicoreDataset):
    def __init__(
        self,
        data_path,
        vocab_path,
        seq_max_len: int = 1201,
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        pad_value: int = -2,
        cls_appended: bool = True,
        use_gnn: bool = True,
        use_embed: bool = False,
        text_emb_path: str = None,
        edge_corr_thr: int = 5,
        split: str = "train",
        mode: str = "train",
        shuffle: bool = True,
        preprocess: bool = True,
    ):
        super().__init__()
        self.epoch = 0
        self.lmdb_path = os.path.join(data_path,split)
        self.vocab_path = vocab_path
        self.seq_max_len = seq_max_len
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.pad_value = pad_value
        self.cls_appended = cls_appended
        self.use_gnn = use_gnn
        self.edge_corr_thr = edge_corr_thr
        self.split = split
        self.mode = mode
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.use_embed = use_embed
        if self.use_embed:
            with open(text_emb_path,"rb") as f:
                self.GeneIDEmbeddingSer = pickle.load(f)
        else:
            self.GeneIDEmbeddingSer = None
        self.set_epoch(1)
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        self.lib = self.env.begin(write=False)
        self.vocab = GeneVocab.from_file(self.vocab_path)
        if self.lib.get('sample_num'.encode()) is not None:
            self.data_len = pickle.loads(self.lib.get('sample_num'.encode()))
            self.n_cls = pickle.loads(self.lib.get('celltype_num'.encode()))
            celltypes = pickle.loads(self.lib.get('celltype_set'.encode()))
            self.celltype_to_id = celltypes
            self.batch_num = pickle.loads(self.lib.get('batch_num'.encode()))
            logger.info(f"[*]Successfully load lmdb {self.split} dataset!")
            logger.info(f"[*]Total number of samples is {self.data_len}")
            logger.info(f"[*]The size of gene vocab is {len(self.vocab)}")
            logger.info(f"[*]Whether use graph data: {self.use_gnn}")          
        else:
            self.data_len = pickle.loads(self.lib.get('sample_number'.encode()))
            print(f"[***]open lmdb database")
            logger.info(f"[*]Successfully load lmdb {self.split} dataset!")
            logger.info(f"[*]Total number of samples is {self.data_len}")
            logger.info(f"[*]The size of gene vocab is {len(self.vocab)}")
            logger.info(f"[*]Whether use graph data: {self.use_gnn}")
        torch.multiprocessing.set_start_method("fork",force=True)

    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()

    def __len__(self):
        return self.data_len

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        res = np.arange(len(self), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(res)
        return res

    def __getitem__(self, idx):
        data_dict = pickle.loads(self.lib.get(str(idx).encode()))
        data_dict['gene_list'] = data_dict['gene_list'].to(torch.int)
        data_dict['gene_list'] = data_dict['gene_list'] + torch.where(data_dict['gene_list']<0,65536,0)
        if self.use_embed:
            UsedGeneIDEmbeddingSer = self.GeneIDEmbeddingSer.reindex(pd.Series(data_dict['gene_list']))
            data_dict['gene_summary_embed'] = torch.from_numpy(np.vstack([embedding for embedding in UsedGeneIDEmbeddingSer]))
        else:
            data_dict['gene_summary_embed'] = None
        if self.use_gnn==False:
            data_dict['edge_index']=None
        else:
            nnode=len(data_dict['gene_list'])
            ratio=4
            if nnode*ratio<data_dict['edge_index'].shape[1]:
                if sum(data_dict['edge_attr_type'][1]==1).item()>nnode*ratio:
                    res_idx = torch.where((data_dict['edge_attr_type'][0]==1)|(data_dict['edge_attr_type'][2]==1))[0]
                    cls_idx = torch.where((data_dict['edge_index'][1]==0))[0]
                    sub_idx = data_dict['edge_attr_type'][1]*abs(data_dict['edge_attr_corr'])
                    sample_idx = torch.multinomial(sub_idx.to(torch.float32),nnode*ratio, replacement=False)
                    ret_idx = torch.cat([res_idx,cls_idx,sample_idx])
                    ret_idx = torch.unique(ret_idx)
                    # import pdb;pdb.set_trace()
                    data_dict['edge_attr_corr']=torch.index_select(data_dict['edge_attr_corr'],0,ret_idx)
                    data_dict['edge_attr_chromo']=torch.index_select(data_dict['edge_attr_chromo'],0,ret_idx)
                    data_dict['edge_attr_type']=torch.index_select(data_dict['edge_attr_type'][0],0,ret_idx)
                    data_dict['edge_index']=torch.index_select(data_dict['edge_index'],1,ret_idx)
                else:
                    data_dict['edge_attr_type']=data_dict['edge_attr_type'][0] 
            else:
                data_dict['edge_attr_type']=data_dict['edge_attr_type'][0]
        return data_dict

    @staticmethod
    def collater(samples):
        # TODO modify batch dict for older version

        samples = [s for s in samples if s is not None]

        if len(samples) == 0:
            return None

        batch = {}
        batch['gene_list'] = torch.stack([s['gene_list'] for s in samples], dim=0)
        batch['values'] = torch.stack([s['values'] for s in samples], dim=0)
        batch['truth'] = batch['values'].clone()
        batch['batch_id'] = torch.tensor([s['batch_id'] for s in samples], dtype=torch.int)
        batch['celltype'] = torch.tensor([s['celltype'] for s in samples], dtype=torch.int)
        batch['idx'] = torch.tensor([s['idx'] for s in samples])
        if samples[0]['gene_summary_embed'] is None:
            batch['gene_embed'] = None
        else:
            batch['gene_embed']=torch.stack([s['gene_summary_embed'] for s in samples], dim=0)
        if samples[0]['edge_index'] is None:
            batch['edge_index'] = None
            batch['edge_attr'] = None
        else:
            edge_index = [s['edge_index'].to(torch.long) for s in samples]
            nnode = batch['gene_list'].shape[1]
            batch['edge_index'] = torch.cat(
                [edge_index[idx] + nnode * idx for idx in range(len(edge_index))],
                dim = 1
            )
            batch['edge_attr_tftg'] = torch.cat([s['edge_attr_type'] for s in samples])
            batch['edge_attr_corr'] = torch.cat([s['edge_attr_corr'] for s in samples])
            batch['edge_attr_chromo'] = torch.cat([s['edge_attr_chromo'] for s in samples])
            batch['edge_attr'] = torch.vstack((batch['edge_attr_tftg'],batch['edge_attr_corr'],batch['edge_attr_chromo']))
        return batch
