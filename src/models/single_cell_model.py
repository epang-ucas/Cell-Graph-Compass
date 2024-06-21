import lmdb, pickle, os
import logging
import numpy as np
from typing import Any, Union
import torch

from unicore.models import BaseUnicoreModel, register_model, register_model_architecture

from .gene_model import GeneModel
from .gene_tokenizer import GeneVocab
logger = logging.getLogger(__name__)


@register_model("single_cell_model")
class ScModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--data_path", type=str, default=None)
        parser.add_argument("--vocab_path", type=str, default=None)
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument("--d_hid", type=int, default=512)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--nlayers", type=int, default=12)
        parser.add_argument("--n_bins", type=int, default=51)

        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--pad_token", type=str, default="<pad>")
        parser.add_argument("--pad_value", type=float, default=-2)
        parser.add_argument("--mask_ratio", type=float, default=0.4)
        parser.add_argument("--mask_value", type=float, default=-3)

        parser.add_argument("--do_mvg", action="store_true", default=False)
        parser.add_argument("--do_mvc", action="store_true", default=False)
        parser.add_argument("--do_dab", action="store_true", default=False)
        parser.add_argument("--do_cce", action="store_true", default=False)
        parser.add_argument("--dab_weight", type=float, default=1.0)
        parser.add_argument("--do_cls", action="store_true", default=False)
        parser.add_argument("--cls_weight", type=float, default=1.0)
        parser.add_argument("--use_batch_labels", action="store_true", default=False)
        parser.add_argument("--domain_spec_batchnorm", action="store_true", default=False)
        parser.add_argument("--do_ecs", action="store_true", default=False)
        parser.add_argument("--ecs_threshold", type=float, default=0.8)
        parser.add_argument("--explicit_zero_prob", action="store_true", default=False)
        parser.add_argument("--pre_norm", type=bool, default=False)
        parser.add_argument("--amp", type=bool, default=True)
        parser.add_argument("--cell_emb_style", type=str, default="cls")
        parser.add_argument("--use_detach", action="store_true", default=False)

        parser.add_argument("--use_gnn", action="store_true", default=False)
        parser.add_argument("--nlayers_gnn", type=int, default=3)
        parser.add_argument("--n_message", type=int, default=1)
        parser.add_argument("--n_edge_layers", type=int, default=1)
        parser.add_argument("--use_fast_transformer", action="store_true", default=False)
        parser.add_argument("--fast_transformer_backend", type=str, default="flash")

        # parser.add_argument("--from_scatch", action="store_true", default=False)
        parser.add_argument("--pretrain", type=str, default=None)
        
        
    def __init__(self, args):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.pad_value = args.pad_value
        self.mask_ratio = args.mask_ratio
        self.mask_value = args.mask_value
        self.explicit_zero_prob = args.explicit_zero_prob
        self.do_mvg = args.do_mvg
        self.do_mvc = args.do_mvc
        self.ecs_thres = args.ecs_threshold
        self.do_dab = args.do_dab
        self.dab_weight = args.dab_weight
        self.cls_weight = args.cls_weight
        self.do_cls = args.do_cls
        self.do_ecs = args.do_ecs
        self.do_cce = args.do_cce
        self.n_bins = args.n_bins
        vocab_path = os.path.join(args.vocab_path, "vocab.json")
        if os.path.exists(vocab_path):
            vocab = GeneVocab.from_file(vocab_path)
            ntoken = len(vocab)
            padding_idx = vocab[args.pad_token]
        else:
            #TODO: consider situdation when vocab is not offered
            padding_idx = 0
        print(args.data_path)
        if args.do_cls or args.domain_spec_batchnorm or args.use_batch_labels:
            lmdb_path = os.path.join(args.data_path,"train")
            self.env = env = lmdb.open(lmdb_path, readonly=True, lock=False)
            lib = env.begin(write=False)
            self.n_cls = n_cls = pickle.loads(lib.get('celltype_num'.encode()))
            num_batch_labels = pickle.loads(lib.get('batch_num'.encode()))
        else:
            self.n_cls = n_cls = None
            num_batch_labels = None
        self.mvc_decoder_style = "inner product, detach" if args.use_detach else "inner product"

        logger.info(
            f"MVC: {self.do_mvc}, CLS: {self.do_cls}, CCE: {self.do_cce},  \
            ECS: {self.do_ecs}, DAB: {self.do_dab}, do_sample: {False}, \
            explicit_zero_prob: {self.explicit_zero_prob}, \
            use detach: {self.mvc_decoder_style}, \
            ecs:  {self.ecs_thres}"       
        )
        self.model = GeneModel(
            ntoken = ntoken,
            d_model = args.d_model,
            nhead = args.nhead,
            d_hid = args.d_hid,
            nlayers = args.nlayers,
            n_cls = n_cls,
            dropout = args.dropout,
            padding_idx = padding_idx,
            pad_token = args.pad_token,
            pad_value = args.pad_value,
            do_mvg = args.do_mvg,
            do_mvc = args.do_mvc,
            do_cce = args.do_cce,
            do_ecs = args.do_ecs,
            do_dab = args.do_dab,
            do_cls = args.do_cls,
            do_sample = False,
            use_batch_labels = args.use_batch_labels,
            num_batch_labels = num_batch_labels,
            domain_spec_batchnorm = args.domain_spec_batchnorm,
            input_emb_style = "continuous",
            cell_emb_style = args.cell_emb_style,
            mvc_decoder_style = self.mvc_decoder_style,
            ecs_threshold = args.ecs_threshold,
            explicit_zero_prob = args.explicit_zero_prob,
            pre_norm = args.pre_norm,
            use_gnn = args.use_gnn,
            use_embed = args.embed,
            nlayers_gnn = args.nlayers_gnn,
            n_message = args.n_message,
            n_edge_layers = args.n_edge_layers,
            use_fast_transformer = args.use_fast_transformer,
            fast_transformer_backend  = args.fast_transformer_backend,
        )
        self.dtype = torch.float
        if args.pretrain is not None:
            pretrained_dict = torch.load(args.pretrain)['model']
            model_dict = self.state_dict()
            load_dict = dict()
            for key in model_dict.keys():
                if key not in pretrained_dict:
                    logger.info(f"[*]model param: {key} not in pretrain model")
                else:
                    if key.split(".")[1] == "dsbn":
                        logger.info(f"[*]model param: {key} will be trained from scatch")
                        continue
                    if pretrained_dict[key].shape != model_dict[key].shape:
                        logger.info(f"[*]model param: {key} shape is not consistent with pretrain model")
                        logger.info(f"[*]model shape: {model_dict[key].shape}, pretrain model shape: {pretrained_dict[key].shape}")
                    else:
                        load_dict[key] = pretrained_dict[key]
            model_dict.update(load_dict)
            self.load_state_dict(model_dict)
            logger.info("[**]Load model successfully!")
        else:
            logger.info("[**]Train model from scatch!")


    def __del__(self):
        if hasattr(self,'env'):
            self.env.close()

    def half(self):
        self.model = self.model.half()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        self.model = self.model.bfloat16()
        self.dtype = torch.bfloat16
        return self

    def float(self):
        self.model = self.model.float()
        self.dtype = torch.float
        return self

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def random_mask_value(
        self,
        values: torch.Tensor,
        mask_ratio: float = 0.15,
        mask_value: int = -3,
        pad_value: int = -2,
        append_cls: bool = True,
    ) -> torch.Tensor:
        """
        Randomly mask a batch of data.

        Args:
            values (array-like):
                A batch of tokenized data, with shape (batch_size, n_features).
            mask_ratio (float): The ratio of genes to mask, default to 0.15.
            mask_value (int): The value to mask with, default to -1.
            pad_value (int): The value of padding in the values, will be kept unchanged.

        Returns:
            torch.Tensor: A tensor of masked data.
        """
        masked_value = values.clone().detach()
        mask = torch.zeros_like(masked_value)
        if not append_cls:
            row = masked_value
        else:
            row = masked_value[:,1:]
        res = row - pad_value
        res[res!=0]=1
        if not append_cls:
            for idx,i in enumerate(res):
                n_mask = int((i!=0).sum().item() * mask_ratio)
                if n_mask==0:
                    continue
                mask_idx = torch.multinomial(i, n_mask, replacement=False)
                mask[idx][mask_idx] = 1
        else:
            for idx,i in enumerate(res):
                n_mask = int((i!=0).sum().item() * mask_ratio)
                if n_mask==0:
                    continue
                mask_idx = torch.multinomial(i, n_mask, replacement=False) + 1
                mask[idx][mask_idx] = 1
        mask = mask.bool()
        masked_value[mask] = mask_value
        return masked_value, mask

    def forward(self, batch, **kwargs):
        batch['values'] = batch['values'].to(self.dtype)
        batch['truth'] = batch['truth'].to(self.dtype)
        if batch['gene_embed'] is not None:
            batch['gene_embed'] = batch['gene_embed'].to(self.dtype)
        if batch['edge_index'] is not None:
            batch['edge_attr'] = batch['edge_attr'].to(self.dtype)
        else:
            batch['edge_attr'] = None
        if self.mask_ratio == 0:
            masked_value = batch['values']
            mask = torch.zeros_like(masked_value).to(masked_value)
        else:
            masked_value, mask = self.random_mask_value(
                values = batch['values'],
                mask_ratio = self.mask_ratio,
                mask_value = self.mask_value,
                pad_value = self.pad_value
            )
        src_key_padding_mask = batch['values'].eq(self.pad_value)
        outputs = self.model(
            src = batch['gene_list'],
            values = masked_value,
            embed = batch['gene_embed'],
            src_key_padding_mask = src_key_padding_mask,
            batch_labels = batch['batch_id'],
            edge_index = batch['edge_index'],
            edge_attr = batch['edge_attr'],
        )
        outputs["mask_position"] = mask
        return outputs


@register_model_architecture("single_cell_model", "single_cell_model")
def base_architecture(args):
    args.model_name = getattr(args, "mask_ratio", 0.5)



