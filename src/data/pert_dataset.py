import os, pickle, logging 
import numpy as np
import pandas as pd
from typing import *
# from line_profiler import LineProfiler
import lmdb
import torch
import torch.nn.functional as F

from unicore.data import UnicoreDataset

from models.gene_tokenizer import GeneVocab

torch.multiprocessing.set_start_method("fork", force=True)

logger = logging.getLogger(__file__)

class PertDataset(UnicoreDataset):
    def __init__(
        self,
        data_path,
        vocab_path,
        use_gnn: bool = False,
        split: str = "train",
        mode: str = "train",
        shuffle: bool = True,
        use_embed: bool = True,
        text_emb_path: str = None,
        edge_path: str = None,
    ):
        super().__init__()
        self.epoch = 0
        self.use_gnn = use_gnn 
        self.split = split
        self.mode = mode
        self.shuffle = shuffle

        lmdb_path = os.path.join(data_path, split)
        if not os.path.exists(lmdb_path):
            raise ValueError(f"Wrong lmdb path: {lmdb_path}")
        if not os.path.exists(vocab_path):
            raise ValueError(f"Wrong vocab path: {vocab_path}")
        self.env = env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.lib = lib = env.begin(write=False)
        self.vocab = vocab = GeneVocab.from_file(vocab_path)
        
        self.gene_names = pickle.loads(lib.get('gene_names'.encode()))
        self.node_map = {x: it for it, x in enumerate(self.gene_names)}
        self.token_len = len(self.gene_names)
        gene_vocab_id = [
            vocab[gene] if gene in vocab else vocab["<pad>"] for gene in self.gene_names
        ]
        gene_ids_in_vocab = [
            1 if gene in vocab else -1 for gene in self.gene_names
        ]
        self.gene_vocab_id = torch.tensor(gene_vocab_id)
        self.gene_ids_in_vocab = np.array(gene_ids_in_vocab)
        self.data_len = pickle.loads(lib.get('sample_number'.encode()))
        logger.info(f"[*]Successfully load dataset from {lmdb_path}!")
        logger.info(f"[*]Total number of genes is {self.token_len}")
        logger.info(f"[*]Total number of samples is {self.data_len}")
        logger.info(f"[*]Successfully load vocabuary from {vocab_path}!")
        logger.info(
            f"match {np.sum(self.gene_ids_in_vocab >= 0)}/{len(self.gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(self.vocab)}."
        )
        logger.info(f"[*]Whether use graph data: {self.use_gnn}")
        self.use_embed = use_embed
        if self.use_embed:
            with open(text_emb_path,"rb") as f:
                self.GeneIDEmbeddingSer = pickle.load(f)
        else:
            self.GeneIDEmbeddingSer = None
        if use_gnn:
            with open(edge_path,"rb") as f:
                index_edge,type_attr,corr_attr,chromo_attr = pickle.load(f)
            self.edge_index = index_edge
            self.attr_type = type_attr
            self.attr_corr = corr_attr
            self.attr_chromo = chromo_attr
        else:
            self.edge_index = self.edge_attr = None
        self.vocab.set_default_index(vocab["<pad>"])
        self.set_epoch(1)
        
        

    def __del__(self):
        self.env.close()

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        data_dict = pickle.loads(self.lib.get(str(idx).encode()))
        values = data_dict['values']
        target = data_dict['target']
        if self.use_embed:
            UsedGeneIDEmbeddingSer = self.GeneIDEmbeddingSer.reindex(pd.Series(self.gene_vocab_id))
            gene_summary_embed = torch.from_numpy(np.vstack([embedding for embedding in UsedGeneIDEmbeddingSer]))
        else:
            gene_summary_embed= None
            
        if self.use_gnn==True:
            nnode=len(data_dict['values'])
            ratio=4
            if nnode*ratio<self.edge_index.shape[1] and sum(self.attr_type[1]==1).item()>nnode*ratio:
                res_idx = torch.where((self.attr_type[0]==1)|(self.attr_type[2]==1))[0]
                cls_idx = torch.where((self.edge_index[1]==0))[0]
                sub_idx = self.attr_type[1]*abs(self.attr_corr)
                sample_idx = torch.multinomial(sub_idx.to(torch.float32),nnode*ratio, replacement=False)
                ret_idx = torch.cat([res_idx,cls_idx,sample_idx])
                ret_idx = torch.unique(ret_idx)
                edge_attr_corr=torch.index_select(self.attr_corr,0,ret_idx)
                edge_attr_chromo=torch.index_select(self.attr_chromo,0,ret_idx)
                edge_attr_tftg=torch.index_select(self.attr_type[0],0,ret_idx)
                edge_index=torch.index_select(self.edge_index,1,ret_idx)
                edge_attr=torch.vstack((edge_attr_tftg,edge_attr_corr,edge_attr_chromo))
            else:
                edge_index=self.edge_index
                edge_attr=torch.vstack((self.attr_type[0],self.attr_corr,self.attr_chromo))
        else:
            edge_index = None
            edge_attr = None
        feature_dict = {
            'genes': self.gene_vocab_id,
            'values': values, 
            'target': target, 
            'pert': data_dict['pert'],
            'gene_summary_embed': gene_summary_embed,
            'de_idx': torch.tensor(data_dict['de_idx']),
            'pert_name': data_dict['pert_name'],
            'edge_index': edge_index,
            'edge_attr': edge_attr,
        }
        return feature_dict

    @staticmethod
    def collater(samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None

        batch = {}
        batch['gene_list'] = torch.stack([s['genes'] for s in samples], dim=0)
        batch['values'] = torch.stack([s['values'] for s in samples], dim=0)
        batch['target'] = torch.stack([s['target'] for s in samples], dim=0)
        batch['pert'] = torch.stack([s['pert'] for s in samples], dim=0)
        batch['de_idx'] = torch.stack([s['de_idx'] for s in samples], dim=0)
        batch['pert_name'] = [s['pert_name'] for s in samples]
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
            batch['edge_attr'] = torch.cat([s['edge_attr'] for s in samples], dim=1)
        return batch
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        res = np.arange(len(self), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(res)
        return res
