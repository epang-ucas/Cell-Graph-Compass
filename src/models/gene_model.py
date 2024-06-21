import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.utils import add_self_loops, degree
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.distributions import Bernoulli
from tqdm import trange

import numpy as np
import copy
from typing import Dict, Mapping, Optional, Tuple, Any, Union

from .gene_gnn import GeneConvLayer

from flash_attn.flash_attention import FlashMHA
from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse

class GeneModel(nn.Module):

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int = 3,
        n_cls: Optional[int] = None,
        dropout: float = 0.2,
        padding_idx: int = 0,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvg: bool = False,
        do_mvc: bool = False,
        do_cce: bool = False,
        do_ecs: bool = False,
        do_dab: bool = False,
        do_cls: bool = False,
        do_sample: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = "continuous",
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        pre_norm: bool = False,

        use_gnn: bool = False,
        use_embed: bool = False,
        nlayers_gnn: int = 3,
        n_message: int = 3,
        n_edge_layers: int = 3,

        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
    ):
        super().__init__()
        if use_fast_transformer:
            self.model_type = "GNN+fastTransformer" if use_gnn else "fastTransformer"
        else:
            self.model_type = "GNN+Transformer" if use_gnn else "Transformer"
        self.d_model = d_model
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"

        self.do_mvg = do_mvg
        self.do_mvc = do_mvc
        self.do_cce = do_cce
        self.do_ecs = do_ecs
        self.do_dab = do_dab
        self.do_cls = do_cls
        self.do_sample = do_sample
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.use_gnn = use_gnn
        self.use_embed = use_embed
        self.dtype = torch.float

        ## Encoder
        # Gene Tokens Encoder
        self.gene_encoder = GeneEncoder(ntoken, d_model, padding_idx=padding_idx)  # tmp padding_idx=vocab[pad_token]
        # Gene Expression Encoder
        self.value_encoder = ValueEncoder(d_model, dropout)
        # Batch Id Encoder
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)
        # Domain Specific Batch_norm
        if domain_spec_batchnorm:
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        else:
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)
        # GNN Encoder
        if use_gnn:
            self.edge_encoder = EdgeEncoder(d_model, dropout=dropout)
            self.gnn_encoder = GnnEncoder(
                d_model,
                d_model,
                n_message = n_message,
                n_edge_layers = n_edge_layers,
                num_encoder_layers = nlayers_gnn,
                drop_rate = dropout,
            )
        if use_embed:
            self.gene_summary_encoder = GeneSummaryEncoder(input_size=768, output_size=d_model)
        # Transformer Encoder
        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = nn.TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=False)

        ## Decoder
        # Gene Expression Predicition 
        if self.do_mvg:
            self.decoder = ExprDecoder(
                d_model,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )
        # Genetype Classification 
        if self.do_cls:
            self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        # Gene Expression Predicition with Cell Embedding 
        if self.do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )
        # Domain Adaptation with Reverse Back-propagation 
        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
        # Elastic Cell Similarity (ECS)
        if self.do_cce:
            self.sim = Similarity(temp=0.5)
            self.creterion_cce = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.gene_encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        embed: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,  # (batch,)
        edge_index: Tensor = None,
        edge_attr: Tensor = None,
    ) -> Tensor:
        self._check_batch_labels(batch_labels)
        src = self.gene_encoder(src)  # (batch, seq_len, embsize)
        self.gene_embedding = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            if self.use_embed:
                summary_embed = self.gene_summary_encoder(embed)
                total_embs = src + values + summary_embed
            else:
                total_embs = src + values

        if self.domain_spec_batchnorm:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        else:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_gnn:
            if edge_index is None:
                raise ValueError("Using GNN should provide edge_index!")
            if edge_attr is None:
                raise ValueError("Using GNN should provide edge_attr!")
            edge_attr = self.edge_encoder(edge_attr)
            total_embs = self.gnn_encoder(total_embs, edge_index, edge_attr)
        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output[:, 1:, :], dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        # elif batch_labels is not None:
        #     raise ValueError(
        #         "batch_labels should only be provided when `self.use_batch_labels`"
        #         " or `self.domain_spec_batchnorm` is True"
        #     )

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        embed: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encode(
            src,
            values,
            embed,
            src_key_padding_mask, 
            batch_labels, 
            edge_index, 
            edge_attr
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
        output = {}
        if self.do_mvg:
            mlm_output = self.decoder(
                transformer_output
                if not self.use_batch_labels
                else torch.cat(
                    [
                        transformer_output,
                        batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                    ],
                    dim=2,
                ),
                # else transformer_output + batch_emb.unsqueeze(1),
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
                output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
            else:
                output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb
        if self.do_cls:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if self.do_cce:
            cell1 = cell_emb
            transformer_output2 = self._encode(
                src, values, embed, src_key_padding_mask, batch_labels, edge_index
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        if self.do_mvc:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.gene_embedding,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if self.do_ecs:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)
            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)
            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)
        return output

    def encode_batch(
        self,
        dataset,
        batch_size: int,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
        sample_rate: float = 0.7,
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        data_len = dataset.data_len
        device = next(self.parameters()).device

        outputs = list()
        batch_ids = list()
        celltypes = list()
        values = list()
        cls_outputs = list()

        for i in trange(0, data_len, batch_size):
            data_batch = [
                dataset.__getitem__(idx) 
                for idx in range(i, i+batch_size) if idx < data_len
            ]
            batch = dataset.collater(data_batch)
            if not data_batch:
                print(f"[*****]{i}")
            # batch = dict()
            batch = {
                k: v.to(device) if v is not None else None 
                for k, v in batch.items()
            }

            src_key_padding_mask = batch["gene_list"].eq(dataset.vocab[dataset.pad_token])
            raw_output = self._encode(
                src = batch["gene_list"],
                values = batch['values'].to(self.dtype),
                embed = batch["gene_embed"],
                src_key_padding_mask = src_key_padding_mask,
                batch_labels = batch["batch_id"],
                edge_index = batch["edge_index"],
                edge_attr = batch["edge_attr"].to(self.dtype) 
                if batch["edge_attr"] is not None else None,
            )
            batch_ids.append(batch["batch_id"].detach().cpu())
            celltypes.append(batch["celltype"].detach().cpu())
            values.append(batch['values'].detach().cpu())
            output = raw_output.detach().cpu()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs.append(output)
            if self.do_cls:
                cls_output = self.cls_decoder(raw_output[:, time_step, :])
                cls_outputs.append(cls_output.detach().cpu())
        batch_ids = torch.concat(batch_ids, dim=0)
        celltypes = torch.concat(celltypes, dim=0)
        values = torch.concat(values, dim=0)
        outputs = torch.concat(outputs, dim=0)
        cls_outputs = torch.concat(cls_outputs, dim=0)
        if return_np:
            batch_ids = batch_ids.cpu().numpy()
            celltypes = celltypes.cpu().numpy()
            values = values.cpu().numpy()
            outputs = outputs.cpu().numpy()
            cls_outputs = cls_outputs.cpu().numpy()
        return outputs, batch_ids, celltypes, values, cls_outputs

class FastTransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float
    ) -> nn.Module:

        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, "
                f"got d_model={d_model} and nhead={nhead}"
            )
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    @staticmethod
    def build_length_mask(
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> "LengthMask":

        from fast_transformers.masking import LengthMask

        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings  # (N,)
        length_mask = LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

        if src_key_padding_mask[length_mask.bool_matrix].sum() != 0:
            raise ValueError(
                "Found padding tokens in the middle of the sequence. "
                "src_key_padding_mask and length_mask are not compatible."
            )
        return length_mask

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len, embsize]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        if src_key_padding_mask.shape != src.shape[:2]:
            raise ValueError(
                f"src_key_padding_mask shape {src_key_padding_mask.shape} "
                f"does not match first two dims of src shape {src.shape[:2]}"
            )

        if src_key_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"src_key_padding_mask needs to be of type torch.bool, "
                f"got {src_key_padding_mask.dtype}"
            )

        length_mask = self.build_length_mask(src, src_key_padding_mask)
        output = self.fast_transformer_encoder(src, length_mask=length_mask)
        return output


class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src

class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.norm(x)
        return x

class ValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(
        self, 
        d_model: int, 
        dropout: float = 0.1, 
        max_value: int = 512
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)

class GeneSummaryEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(GeneSummaryEncoder, self).__init__()
        
        self.layer = nn.Linear(input_size, output_size)
        # self.input_layer = nn.Linear(input_size, hidden_layers[0])
        # self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
        # self.output_layer = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):
        x = F.relu(self.layer(x))
        # for hidden_layer in self.hidden_layers:
            # x = F.relu(hidden_layer(x))
        # x = self.output_layer(x)
        return x

class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.norm(x)
        return x




class EdgeEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        max_dis: int=50,
    ) -> None:
        super().__init__()
        self.tftg_embedding = nn.Sequential(
            nn.Embedding(2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.corr_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(p=dropout),
        )
        self.chromo_embedding = nn.Sequential(
            nn.Embedding(2*max_dis+2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(
            self, 
            edge_attr: Tensor,
    ) -> Tensor:
        """
        Args:
            edge_attr: Tensor, shape [n_edge, 5]  
        """
        edge_tftg = self.tftg_embedding(edge_attr[0].to(torch.int))
        edge_corr = self.corr_embedding(edge_attr[1].unsqueeze(-1))
        edge_chromosome = self.chromo_embedding(edge_attr[2].to(torch.int))
        edge_embed =edge_tftg + edge_corr + edge_chromosome # [n_edege, embedding_dim]
        return edge_embed


class GnnEncoder(nn.Module):
    def __init__(
        self,
        node_hidden_dim: int,
        edge_hidden_dim: int,
        n_message: int = 3,
        n_edge_layers: int = 3,
        num_encoder_layers: int = 2,
        drop_rate: float = 0.1, 
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
                GeneConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate = drop_rate,
                    n_message = n_message,
                    n_edge_gvps = n_edge_layers,
                    layernorm = True,
                ) 
            for _ in range(num_encoder_layers)
        )

    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        '''
        :param x: node embedding of shape [bsz, n_node, node_dims]
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: edge embedding of shape [n_edges, edge_dims] 
        '''    
        # x = self.activation(self.gnn1(x, edge_index)) 
        # x = self.gnn2(x, edge_index)
        # x = self.norm(x)
        node_embeddings = torch.flatten(x, 0, 1)  # [bsz*n_node, node_dims]
        edge_embeddings = edge_attr
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings, edge_index, edge_embeddings)
        node_embeddings = torch.unflatten(node_embeddings, 0, (x.shape[0], x.shape[1]))
        return node_embeddings

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        else:
            zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)

class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.

    There are actually three ways of making this, all start with gene_embs -> query_vecs,
    and then:
    1. cell_emb x W x query vecs.
       This one makes the most sense, since in the query space, the query look at
       different dimensions of cel_emb and sync them. This one has explicit interaction.
    2. FC([cell_emb, query_vecs]).
       This one has the benifit to have smaller query_vecs and makes them like bottle
       neck layer. For example 64 dims.
    3. FC(cell_emb + query_vecs).

    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)

class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)