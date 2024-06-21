import argparse, copy, os, sys
from typing import List, Tuple, Dict, Union, Optional
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import scib
from anndata import AnnData
import anndata as ad
import scanpy as sc



import torch
import torch.nn as nn
import pickle
import logging
from tqdm import trange
import torchmetrics

from data.single_cell_dataset import ScDataset
from models.single_cell_model import ScModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
    level=logging.INFO
)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter("%(asctime)s - %(message)s")
# ch.setFormatter(formatter)
# logger.addHandler(ch)

def get_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Inference')
    
    parser.add_argument("data_path", default=None) 
    parser.add_argument("best_model", default=None)
    parser.add_argument("save_dir", default=None)
    parser.add_argument("vocab_path", default=None)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--seq_max_len", type=int, default=1201)
    parser.add_argument("--edge_corr_thr", type=int, default=5)
    parser.add_argument("--use_graph", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--data_preprocess", action="store_true", default=False)
    
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_hid", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--n_bins", type=int, default=51)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pad_token", type=str, default="<pad>")
    parser.add_argument("--pad_value", type=float, default=-2)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--mask_value", type=float, default=-1)

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
    parser.add_argument("--ecs_threshold", type=float, default=0.6)
    parser.add_argument("--explicit_zero_prob", action="store_true", default=False)
    parser.add_argument("--pre_norm", type=bool, default=False)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--cell_emb_style", type=str, default="cls")

    parser.add_argument("--use_gnn", action="store_true", default=False)
    parser.add_argument("--nlayers_gnn", type=int, default=3)
    parser.add_argument("--n_message", type=int, default=1)
    parser.add_argument("--n_edge_layers", type=int, default=1)
    parser.add_argument("--use_myTransformer", action="store_true", default=False)
    parser.add_argument("--use_fast_transformer", action="store_true", default=False)
    parser.add_argument("--use_detach", action="store_true", default=False)
    parser.add_argument("--embed", action="store_true", default=False)
    parser.add_argument("--fast_transformer_backend", type=str, default="flash")
    parser.add_argument("--pretrain", type=str, default=None)

    parser.add_argument("--text_emb_path", default=None)

    args = parser.parse_args()
    return args

# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
def eval_scib_metrics(
    adata: AnnData,
    embed: str,
    batch_key: str = "str_batch",
    label_key: str = "celltype",
    notes: Optional[str] = None,
) -> Dict:
    # results = scib.metrics.metrics(
    #     adata,
    #     adata_int=adata,
    #     batch_key=batch_key,
    #     label_key=label_key,
    #     embed=embed,
    #     isolated_labels_asw_=True,
    #     silhouette_=True,
    #     hvg_score_=True,
    #     graph_conn_=True,
    #     pcr_=True,
    #     isolated_labels_f1_=True,
    #     trajectory_=True,
    #     nmi_=True,  # use the clustering, bias to the best matching
    #     ari_=True,  # use the clustering, bias to the best matching
    #     cell_cycle_=True,
    #     kBET_=True,  # kBET return nan sometimes, need to examine
    #     ilisi_=True,
    #     clisi_=True,
    # )
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed=embed,
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_score_=0,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=1,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
        organism='human',
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )
    result_dict["avg_batch"] = np.mean(
        [
            result_dict["ASW_label/batch"],
            result_dict["graph_conn"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}
    print("Cell Clustering metrics results")
    for k, v in result_dict.items():
        print(f"{k}: {v}")

    return result_dict

def eval_testdata(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    args,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    if "cls" in include_types:
        logger.info("[*]Evaluating cls cell embeddings")
        # src_key_padding_mask = data["gene_list"].eq(vocab[args.pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            cell_embeddings, batch_ids, celltypes, values, cls_outputs = model.model.encode_batch(
                dataset,
                batch_size = args.batch_size,
                time_step = 0,
                return_np = True,
            )
        # import pdb;pdb.set_trace()
        cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
        cls_outputs = cls_outputs / np.linalg.norm(cls_outputs, axis=1, keepdims=True)

        obs = pd.DataFrame()
        obs["celltype"] = celltypes
        obs["celltype"] = obs["celltype"].astype("category")
        obs["batch_ids"] = batch_ids.astype("category")
        adata = ad.AnnData(X=values, obs=obs, dtype="float32")
        print(f"cell_embeddings shape:  {cell_embeddings.shape}")
        print(f"cls_outputs shape:  {cls_outputs.shape}")
        adata.obsm["X_cell"] = cls_outputs

        results = {}
        try:
            results = eval_scib_metrics(adata, "X_cell",batch_key='batch_ids',label_key='celltype')
        except Exception as e:
            traceback.print_exc()
            logger.info(e)

        sc.pp.neighbors(adata, use_rep="X_cell")
        sc.tl.umap(adata, min_dist=0.3)
        fig = sc.pl.umap(
            adata,
            color=['batch_ids'],
            title=[f"batch, avg_batch = {results.get('avg_batch', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig

        sc.pp.neighbors(adata, use_rep="X_cell")
        sc.tl.umap(adata, min_dist=0.05)
        fig = sc.pl.umap(
            adata,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig
    
        results["batch_umap"].savefig(
            os.path.join(args.save_dir, f"embeddings_batch_umap[cls]_epoch[test]_train1w.png"), dpi=300
        )

        results["celltype_umap"].savefig(
            os.path.join(args.save_dir, f"embeddings_celltype_umap[cls]_epoch[test]_train1w.png"), dpi=300
        )

    if len(include_types) == 1:
        return results

def eval(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    args,
    checkpoint="best",
    drawgraph = True,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset"""
    model.eval()
    logger.info("[*]Evaluating cls cell embeddings")

    data_len = dataset.data_len
    device = next(model.parameters()).device
    dtype = model.model.dtype
    bsz = args.batch_size
    logger.info(f"[*]model.model.device: {device}")
    logger.info(f"[*]model.model.dtype: {dtype}")
    if model.do_cls:
        acc_macro = torchmetrics.Accuracy(
            task = "multiclass", num_classes = model.n_cls, average = 'macro'
        ).to(device)
        acc_micro = torchmetrics.Accuracy(
            task = "multiclass", num_classes = model.n_cls, average = 'micro'
        ) .to(device)
        confmat_abs = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=model.n_cls
        ).to(device)
        confmat_norm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=model.n_cls, normalize='true'
        ).to(device)

    cell_embeddings, batch_ids, celltypes, values, cls_outputs = list(), list(), list(), list(), list()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
        for i_batch in trange(0, data_len, bsz):
            data_batch = [
                dataset.__getitem__(idx) 
                for idx in range(i_batch, i_batch+bsz) if idx < data_len
            ]
            batch = dataset.collater(data_batch)
            batch = {
                k: v.to(device) if v is not None else None 
                for k, v in batch.items()
            }

            src_key_padding_mask = batch["gene_list"].eq(dataset.vocab[dataset.pad_token])
            if args.embed:
                transformer_output = model.model._encode(
                    src = batch["gene_list"],
                    values = batch['values'].to(dtype),
                    embed = batch['gene_embed'],
                    src_key_padding_mask = src_key_padding_mask,
                    batch_labels = batch["batch_id"],
                    edge_index = batch["edge_index"],
                    edge_attr = batch["edge_attr"].to(dtype) 
                    if batch["edge_attr"] is not None else None,
                )
            else:
                transformer_output = model.model._encode(
                    src = batch["gene_list"],
                    values = batch['values'].to(dtype),
                    embed = None,
                    src_key_padding_mask = src_key_padding_mask,
                    batch_labels = batch["batch_id"],
                    edge_index = batch["edge_index"],
                    edge_attr = batch["edge_attr"].to(dtype) 
                    if batch["edge_attr"] is not None else None,
                )
            cell_embedding = transformer_output[:, 0, :]

            if model.do_cls:
                cls_output = model.model.cls_decoder(cell_embedding)
                acc_macro.update(cls_output, batch["celltype"])
                acc_micro.update(cls_output, batch["celltype"])
                confmat_abs.update(cls_output, batch["celltype"])
                confmat_norm.update(cls_output, batch["celltype"])
                cls_outputs.append(cls_output.detach().cpu())
            batch_ids.append(batch["batch_id"].detach().cpu())
            celltypes.append(batch["celltype"].detach().cpu())
            values.append(batch['values'].detach().cpu())
            cell_embeddings.append(cell_embedding.detach().cpu())

    batch_ids = torch.concat(batch_ids, dim=0).numpy()
    celltypes = torch.concat(celltypes, dim=0).numpy()
    values = torch.concat(values, dim=0).numpy()
    cell_embeddings = torch.concat(cell_embeddings, dim=0).numpy()
    cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    if model.do_cls:
        cls_outputs = torch.concat(cls_outputs, dim=0).numpy()
        val_acc_macro = acc_macro.compute() 
        val_acc_micro = acc_micro.compute()
        cm_abs = confmat_abs.compute().cpu().numpy()
        cm_norm = confmat_norm.compute().cpu().numpy()
        logger.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info(f"[*]Macro accuracy on all data: {val_acc_macro}")
        logger.info(f"[*]Micro accuracy on all data: {val_acc_micro}")
        logger.info(f"[*]Confusion Matrix's diagonal acc: {np.diagonal(cm_abs)}")
        logger.info(f"[*]Confusion Matrix(norm)'s diagonal acc: {np.diagonal(cm_norm)}")
        logger.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        cls_outputs = cls_outputs / np.linalg.norm(cls_outputs, axis=1, keepdims=True)
    else:
        val_acc_macro = 0
        val_acc_micro = 0
        cm_abs = 0
        cm_norm = 0

    obs = pd.DataFrame()
    obs["celltype"] = celltypes
    obs["celltype"] = obs["celltype"].astype("category")
    obs["batch_ids"] = batch_ids
    adata = ad.AnnData(X=values, obs=obs, dtype="float32")
    adata.obsm["X_cell"] = cell_embeddings
    # adata.obsm["Y_cell"] = cls_outputs

    results = {}
    try:
        results = eval_scib_metrics(adata, "X_cell",batch_key='batch_ids',label_key='celltype')
    except Exception as e:
        traceback.print_exc()
        logger.info(e)
    if drawgraph:
        with open(os.path.join(args.save_dir,f"result_{checkpoint}.pickle"),"wb") as f:
            pickle.dump((adata,results),f)
        adata.obs['batch_ids']=adata.obs['batch_ids'].astype("category")
        sc.pp.neighbors(adata, use_rep="X_cell")
        sc.tl.umap(adata, min_dist=0.3)
        fig = sc.pl.umap(
            adata,
            color=['batch_ids'],
            title=[f"batch, avg_batch = {results.get('avg_batch', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )
        results["batch_umap"] = fig

        sc.pp.neighbors(adata, use_rep="X_cell")
        sc.tl.umap(adata, min_dist=0.3)
        fig = sc.pl.umap(
            adata,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )
        results["celltype_umap"] = fig

        results["batch_umap"].savefig(
            os.path.join(args.save_dir, f"batch_umap[cls]_epoch[{checkpoint}].png"), dpi=300
        )
        results["celltype_umap"].savefig(
            os.path.join(args.save_dir, f"celltype_umap[cls]_epoch[{checkpoint}].png"), dpi=300
        )

    return results['avg_batch'],results['avg_bio'],val_acc_macro,val_acc_micro,cm_norm,cm_abs

def testModel(args,best_model,model,dataset,draw=True):
    pretrained_dict = torch.load(os.path.join(best_model))['model']
    check_num = best_model.split("/")[-1].split('.')[0].split('_')[-1]
    try:
        model.load_state_dict(pretrained_dict)
        logger.info(f"[*]Loading all model params from {best_model}")
    except:
        model_dict = model.state_dict()
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
        model.load_state_dict(model_dict)
        logger.info("[**]Load model successfully!")
    model = model.float()
    model.cuda()
    avg_batch,avg_bio,val_acc_macro,val_acc_micro,cm_norm,cm_abs = eval(model, dataset, args, checkpoint=check_num,drawgraph=draw)
    return check_num,avg_batch,avg_bio,val_acc_macro,val_acc_micro,cm_norm,cm_abs,best_model

if __name__ == "__main__":

    args = get_arguments()
    logger.info(f"[**]args: {args}")

    ## prepare model
    model = ScModel(args)
    # prepare data
    dataset = ScDataset(
        data_path = args.data_path,
        vocab_path = args.vocab_path, 
        seq_max_len = args.seq_max_len,
        use_gnn = args.use_graph,
        edge_corr_thr = args.edge_corr_thr,
        split = "all",
        mode = "valid",
        shuffle = args.shuffle,
        preprocess = args.data_preprocess,
        use_embed = args.embed,
        text_emb_path= args.text_emb_path,
    )

    logger.info(f"[**]Successfully load dataset!")
    logger.info(f"[*] The size of dataset is {dataset.__len__()}")
    logger.info(f"[*] The size of gene vocab is {len(dataset.vocab)}")
    logger.info(f"[*] The number of seq-batch is {dataset.batch_num}")
    logger.info(f"[*] The set of celltype is {dataset.celltype_to_id}")

    logger.info("Test begins")
    testModel(args,args.best_model,model,dataset)
