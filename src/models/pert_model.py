import lmdb, pickle, os
import logging
import numpy as np
from typing import Any, Union
import torch

from unicore.models import BaseUnicoreModel, register_model, register_model_architecture

from .gene_pert_model import GeneModel
from .gene_tokenizer import GeneVocab
logger = logging.getLogger(__name__)


# TODO add embed options
@register_model("pert_model")
class PertModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--data_path", type=str, default=None)
        parser.add_argument("--vocab_path", type=str, default=None)
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument("--d_hid", type=int, default=512)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--nlayers", type=int, default=12)

        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--pad_token", type=str, default="<pad>")
        parser.add_argument("--pad_value", type=float, default=-2)

        parser.add_argument("--do_mvg", action="store_true", default=False)
        parser.add_argument("--do_mvc", action="store_true", default=False)
        parser.add_argument("--do_dab", action="store_true", default=False)
        parser.add_argument("--do_cce", action="store_true", default=False)
        parser.add_argument("--dab_weight", type=float, default=1.0)
        parser.add_argument("--do_cls", action="store_true", default=False)
        parser.add_argument("--cls_weight", type=float, default=1.0)
        parser.add_argument("--use_batchnorm", action="store_true", default=False)
        parser.add_argument("--input_emb_style", type=str, default="continuous")
        parser.add_argument("--do_ecs", action="store_true", default=False)
        parser.add_argument("--ecs_threshold", type=float, default=0.6)
        parser.add_argument("--explicit_zero_prob", action="store_true", default=False)
        parser.add_argument("--do_sample", action="store_true", default=False)
        parser.add_argument("--pre_norm", type=bool, default=False)
        parser.add_argument("--amp", type=bool, default=True)
        parser.add_argument("--cell_emb_style", type=str, default="cls")
        parser.add_argument("--mvc_decoder_style", type=str, default="inner product, detach")
        parser.add_argument("--mvg_pred_pos", type=str, default="full")

        parser.add_argument("--use_gnn", action="store_true", default=False)
        parser.add_argument("--nlayers_gnn", type=int, default=3)
        parser.add_argument("--n_message", type=int, default=1)
        parser.add_argument("--n_edge_layers", type=int, default=1)
        parser.add_argument("--use_fast_transformer", action="store_true", default=False)
        parser.add_argument("--transformer_type", type=str, default="torch")
        parser.add_argument("--fast_transformer_backend", type=str, default="flash")

        parser.add_argument("--pretrain", type=str, default=None)
        parser.add_argument("--freeze_pretrain_param", action="store_true", default=False)
        parser.add_argument("--pert_pad_id", type=int, default=2)
        parser.add_argument("--mvg_decoder_style", type=str, default="continuous")
        parser.add_argument("--focal_alpha", type=float, default=0.5)
        parser.add_argument("--focal_gamma", type=float, default=0)
        parser.add_argument("--focal_weight", type=float, default=1.0)
        parser.add_argument("--train_stage", type=str, default=None)
        parser.add_argument("--negative_ratio", type=float, default=0.9)
        parser.add_argument("--five_class", action="store_true", default=False)
        parser.add_argument("--debug_mode", action="store_true", default=False)
        
        
    def __init__(self, args):
        super().__init__()
        base_architecture(args)
        if args.debug_mode:
            args.transformer_type = "torch"
            args.pretrain = None
            args.nlayers = 2
            args.nhead = 4
            args.d_hid = args.d_model = 64
            args.batch_size = 8
        self.args = args
        self.pad_token = args.pad_token
        self.pad_value = args.pad_value
        self.explicit_zero_prob = args.explicit_zero_prob
        self.do_sample = args.do_sample
        self.do_mvg = args.do_mvg
        self.do_mvc = args.do_mvc
        self.ecs_thres = args.ecs_threshold
        self.do_dab = args.do_dab
        self.dab_weight = args.dab_weight
        self.cls_weight = args.cls_weight
        self.do_cls = args.do_cls
        self.do_ecs = args.do_ecs
        self.do_cce = args.do_cce
        self.mvg_pred_pos = args.mvg_pred_pos
        self.input_emb_style = args.input_emb_style
        self.mvg_decoder_style = args.mvg_decoder_style
        self.focal_alpha = args.focal_alpha
        self.focal_gamma = args.focal_gamma
        self.focal_weight = args.focal_weight
        self.train_stage = args.train_stage
        self.negative_ratio = args.negative_ratio
        self.five_class = args.five_class
        vocab_path = args.vocab_path
        if os.path.exists(vocab_path):
            vocab = GeneVocab.from_file(vocab_path)
            ntoken = len(vocab)
            padding_idx = vocab[args.pad_token]
            self.vocab = vocab
        else:
            padding_idx = 0
            raise ValueError(f"Vocab is not loaded successfully!")
        self.model = GeneModel(
            ntoken = ntoken,
            d_model = args.d_model,
            nhead = args.nhead,
            d_hid = args.d_hid,
            nlayers = args.nlayers,
            nlayers_cls = -1,
            n_cls = -1,
            dropout = args.dropout,
            padding_idx = padding_idx,
            pad_token = args.pad_token,
            pad_value = args.pad_value,
            pert_pad_id = args.pert_pad_id,
            do_mvg = True,
            do_mvc = False,
            do_cce = False,
            do_ecs = False,
            do_dab = False,
            do_cls = False,
            do_sample = args.do_sample,
            use_batch_labels = False,
            num_batch_labels = -1,
            use_batchnorm = args.use_batchnorm,
            input_emb_style = args.input_emb_style,
            cell_emb_style = args.cell_emb_style,
            mvc_decoder_style = args.mvc_decoder_style,
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
            mvg_decoder_style = args.mvg_decoder_style,
            is_five_class = self.five_class,
        )
        self.dtype = torch.float
        if args.pretrain is not None:
            module_to_load = ["gene_encoder", "value_encoder", "transformer_encoder",
                             "edge_encoder", "gnn_encoder","gene_summary_encoder"]
            module_new = ["pert_encoder", "bn", "decoder"]

            pretrained_dict = torch.load(args.pretrain)
            model_dict = self.state_dict()
            # import pdb;pdb.set_trace()

            load_dict = dict()
            for key in model_dict.keys():
                if key.split(".")[1] in module_to_load:
                    load_dict[key] = pretrained_dict[key]
                    logger.info(f"[*]Model params: {key} are loaded.")
                elif key.split(".")[1] in module_new:
                    logger.info(f"[&]Model params: {key} will be trained from scratch.")
                else:
                    raise ValueError(f"[*]Unexpected model params: {key}")
            model_dict.update(load_dict)
            self.load_state_dict(model_dict)

            if args.freeze_pretrain_param:
                for name, param in self.model.named_parameters():
                    if name.split(".")[0] in module_to_load:
                        param.requires_grad = False
                        logger.info(f"[*]Model params: {name} are frozen.")

            logger.info("[**]Load model successfully!")
        else:
            logger.info("[**]Train model from scatch!")

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

    def forward(self, batch, **kwargs):
        values = batch['values'].to(self.dtype)
        src_key_padding_mask = batch['gene_list'].eq(self.vocab[self.pad_token])
        if batch['gene_embed'] is not None:
            batch['gene_embed'] = batch['gene_embed'].to(self.dtype)
        if batch['edge_index'] is not None:
            batch['edge_attr'] = batch['edge_attr'].to(self.dtype)
        else:
            batch['edge_attr'] = None
        # src_key_padding_mask = torch.zeros_like(
        #     values, dtype=torch.bool, device=values.device
        # )
        # import pdb;pdb.set_trace()
        outputs = self.model(
            src = batch['gene_list'],
            values = values,
            embed = batch['gene_embed'],
            input_pert_flags = batch['pert'],
            src_key_padding_mask = src_key_padding_mask,
            embedding_mask = src_key_padding_mask,
            edge_index = batch['edge_index'],
            edge_attr = batch['edge_attr'].to(self.dtype)
            if batch['edge_attr'] is not None else None,
        )
        return outputs

    def pred_perturb(
        self,
        batch_data,
        node_map,
        gene_ids=None,
    ) -> torch.Tensor:
        return self.model.pred_perturb(
            batch_data, node_map, "all", gene_ids, True, 
            padding_idx=self.vocab[self.pad_token],
        )
    

@register_model_architecture("pert_model", "pert_model")
def base_architecture(args):
    args.model_name = getattr(args, "mask_ratio", 0.5)



