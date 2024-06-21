import logging
import os

import contextlib
from typing import Optional

import numpy as np
import torch

from ..data.single_cell_dataset import ScDataset
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

@register_task("single_cell_task")
class CellClusterPretrain(UnicoreTask):
    """Task for training masked language models (e.g., BERT)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data_path", default=None)
        parser.add_argument("vocab_path", default="/train14/superbrain/zlhu12/lmdb_2000w_total_new")
        parser.add_argument("--seq_max_len", type=int, default=1201)
        parser.add_argument("--edge_corr_thr", type=int, default=5)
        parser.add_argument("--use_graph", action="store_true", default=False)
        parser.add_argument("--embed", action="store_true", default=False)
        parser.add_argument("--shuffle", action="store_true", default=False)
        parser.add_argument("--data_preprocess", action="store_true", default=False)
        parser.add_argument("--text_emb_path", default=None)



    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed
        self.data_path = args.data_path
        self.vocab_path = args.vocab_path
        self.seq_max_len = args.seq_max_len
        self.edge_corr_thr = args.edge_corr_thr
        self.use_gnn = args.use_graph
        self.shuffle = args.shuffle
        self.preprocess = args.data_preprocess
        self.use_embed = args.embed
        self.text_emb_path = args.text_emb_path

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
            
        dataset = ScDataset(
            data_path = self.data_path,
            vocab_path = self.vocab_path,
            seq_max_len = self.seq_max_len,
            use_gnn = self.use_gnn,
            use_embed = self.use_embed,
            text_emb_path= self.text_emb_path,
            edge_corr_thr = self.edge_corr_thr,
            split = split,
            mode = split,
            shuffle = self.shuffle,
            preprocess = self.preprocess,
        )
       
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        # self.config = model.config
        return model

    def disable_shuffling(self) -> bool:
        return not self.shuffle
    
    
    # def begin_valid_epoch(self, epoch, model):
    #     """Hook function called before the start of each validation epoch."""
    #     if epoch % self.validate_interval_updates == 0:
    #         model.validate = True
    #     else:
    #         model.validate = False