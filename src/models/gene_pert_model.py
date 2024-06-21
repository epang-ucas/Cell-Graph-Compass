import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Bernoulli
from tqdm import trange

import numpy as np
import copy, logging
from typing import Dict, Mapping, Optional, Tuple, Any, Union

from .gene_gnn import GeneConvLayer
from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse
from .gene_model import(
    GeneEncoder,
    GeneSummaryEncoder,
    ValueEncoder,
    EdgeEncoder,
    GnnEncoder,
    ExprDecoder,
    MVCDecoder,
    ClsDecoder,
    Similarity,
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer
)

logger = logging.getLogger(__name__)


# TODO add embed options
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
        use_batchnorm: bool = True,
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

        pert_pad_id: int = 2,
        mvg_decoder_style:str = 'continuous',
        is_five_class: bool = False,
    ):
        super().__init__()
        self.model_type = "pert model"
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
        self.use_batchnorm = use_batchnorm
        self.use_gnn = use_gnn
        self.use_embed = use_embed
        self.dtype = torch.float
        
        self.pad_token_id = padding_idx
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.mvg_decoder_style = mvg_decoder_style

        assert input_emb_style == "continuous"
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        assert do_mvg
        if mvg_decoder_style not in ["continuous", "category"]:
            raise ValueError(f"Unknown mvg_decoder_style: {mvg_decoder_style}")

        ## Encoder
        # Gene Tokens Encoder
        self.gene_encoder = GeneEncoder(ntoken, d_model, padding_idx=padding_idx)  # tmp padding_idx=vocab[pad_token]
        # Gene Expression Encoder
        self.value_encoder = ValueEncoder(d_model, dropout)
        # Pert Flag Encoder
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)

        # Batch_norm
        if self.use_batchnorm:
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)
        # GNN Encoder
        if use_gnn:
            self.model_type = self.model_type + " Gnn"
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
        if is_five_class: assert mvg_decoder_style == "category"
        # Gene Expression Predicition 
        if self.do_mvg:
            if mvg_decoder_style == "category":
                if is_five_class:
                    self.decoder = FiveCategoryDecoder(d_model)
                else:
                    self.decoder = CategoryExprDecoder(d_model)
            else:
                self.decoder = ExprDecoder(
                    d_model,
                    explicit_zero_prob=explicit_zero_prob,
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
        input_pert_flags,
        src_key_padding_mask: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        edge_index: Tensor = None,
        edge_attr: Tensor = None,
    ) -> Tensor:
        src = self.gene_encoder(src)  # (batch, seq_len, embsize)
        self.gene_embedding = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
        
        if self.use_embed:
            summary_embed = self.gene_summary_encoder(embed)
            total_embs = src + values + perts + summary_embed
        else:
            total_embs = src + values + perts
        if self.use_batchnorm:
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

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        embed: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
        Returns:
            dict of output Tensors.
        """
        if self.explicit_zero_prob and not self.do_sample and not self.training:
            self.do_sample = True
            logger.warning("Auto set do_sample to True when model is in eval mode.")
        transformer_output = self._encode(
            src,
            values,
            embed,
            input_pert_flags,
            src_key_padding_mask, 
            embedding_mask,
            edge_index, 
            edge_attr,
        )

        output = {}
        if self.do_mvg:
            mlm_output = self.decoder(transformer_output)
            if self.mvg_decoder_style == 'continuous':
                if self.explicit_zero_prob and self.do_sample:
                    bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
                    output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
                else:
                    output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
                if self.explicit_zero_prob:
                    output["mlm_zero_probs"] = mlm_output["zero_probs"]
            elif self.mvg_decoder_style == 'category':
                output["up_logits"] = mlm_output["up_logits"]
                output["change_logits"] = mlm_output["change_logits"]
            else:
                raise ValueError(f"Unknown mvg_decoder_style: {self.mvg_decoder_style}")

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb
        if self.do_cls:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if self.do_mvc:
            mvc_output = self.mvc_decoder(
                cell_emb,
                self.gene_embedding,
            )
            if self.explicit_zero_prob and self.do_sample:
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
                values = batch["values"].to(self.dtype),
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

    def one_hot_encode(self, labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            if type(label) == list:
                for tx in range(len(label)):
                    one_hot[i,label[tx]] = 1
            else:
                if label >=0:
                    one_hot[i,label] = 1
                else:
                    one_hot[i,label] = 0
        one_hot = torch.tensor(one_hot)
        return one_hot

    def pred_perturb(
        self,
        batch_data,
        node_map,
        include_zero_gene = "batch-wise",
        gene_ids = None,
        amp = True,
        padding_idx: int = 0,
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x.view(batch_size, -1)  # (batch_size, n_genes)
        # pert_flags = x[:, 1].long().view(batch_size, -1)
        my_list = batch_data.pert ##list of pert
        result = []
        for s in my_list:
            temp = []
            for gene in s:
                if gene != 'ctrl':
                    temp.append(node_map[gene])
            result.append(temp)

        pert_flags = self.one_hot_encode(result, ori_gene_values.shape[1]).long()
        pert_flags = pert_flags.to(device) ### size: [batch_size, n_genes]

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = self.map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = torch.zeros_like(
            #     input_values, dtype=torch.bool, device=device
            # )
            src_key_padding_mask = mapped_input_gene_ids.eq(padding_idx)
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = self(
                    src = mapped_input_gene_ids,
                    values = input_values,
                    input_pert_flags = input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    embedding_mask = src_key_padding_mask,
                    edge_index = None,
                    edge_attr = None,
                )
            output_values = output_dict["mlm_output"].float()
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values

    def map_raw_id_to_vocab_id(
        self,
        raw_ids: Union[np.ndarray, torch.Tensor],
        gene_ids: np.ndarray,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Map some raw ids which are indices of the raw gene names to the indices of the

        Args:
            raw_ids: the raw ids to map
            gene_ids: the gene ids to map to
        """
        if isinstance(raw_ids, torch.Tensor):
            device = raw_ids.device
            dtype = raw_ids.dtype
            return_pt = True
            raw_ids = raw_ids.cpu().numpy()
        elif isinstance(raw_ids, np.ndarray):
            return_pt = False
            dtype = raw_ids.dtype
        else:
            raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

        if raw_ids.ndim != 1:
            raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

        if gene_ids.ndim != 1:
            raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

        mapped_ids: np.ndarray = gene_ids[raw_ids]
        assert mapped_ids.shape == raw_ids.shape
        if return_pt:
            return torch.from_numpy(mapped_ids).type(dtype).to(device)
        return mapped_ids.astype(dtype)

class CategoryExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc1 = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        # self.init_weights()

    def init_weights(self) -> None:
        self.fc1[-1].weight.data.zero_()
        self.fc2[-1].bias.data.fill_(0.5)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        up_logits = self.fc1(x).squeeze(-1)  # (batch, seq_len)
        change_logits = self.fc2(x).squeeze(-1)  # (batch, seq_len)
        # up_probs = torch.sigmoid(up_logits)
        # change_probs = torch.sigmoid(change_logits)
        return dict(up_logits=up_logits, change_logits=change_logits)

class FiveCategoryDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc1 = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 4),
        )
        # self.init_weights()

    def init_weights(self) -> None:
        self.fc1[-1].weight.data.zero_()
        self.fc2[-1].bias.data.fill_(0.5)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        change_logits = self.fc1(x).squeeze(-1)  # (batch, seq_len)
        up_logits = self.fc2(x)  # (batch, seq_len)
        # up_probs = torch.sigmoid(up_logits)
        # change_probs = torch.sigmoid(change_logits)
        return dict(up_logits=up_logits, change_logits=change_logits)