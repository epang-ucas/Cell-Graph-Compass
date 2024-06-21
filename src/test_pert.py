import argparse, copy, os, sys
from typing import List, Tuple, Dict, Union, Optional, Any
import pickle, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import scib
from anndata import AnnData
import anndata as ad
import scanpy as sc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr, spearmanr
from scipy import optimize
import gears

from data.pert_dataset import PertDataset
from models.pert_model import PertModel

import torch
import torch.nn as nn

from tqdm import trange
import torchmetrics
from torch_geometric.loader import DataLoader

from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
    level=logging.INFO
)

def get_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Inference')
    
    parser.add_argument("data_path", default=None) 
    parser.add_argument("vocab_path", type=str, default=None)
    parser.add_argument("best_model", default=None)
    parser.add_argument("save_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=128)

    # parser.add_argument("--seq_max_len", type=int, default=1536)
    # parser.add_argument("--edge_corr_thr", type=int, default=5)
    parser.add_argument("--use_graph", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    # parser.add_argument("--sample_mode", type=str, default="no_sample")

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
    parser.add_argument("--pre_norm", type=bool, default=False)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--cell_emb_style", type=str, default="cls")
    parser.add_argument("--mvc_decoder_style", type=str, default="inner product, detach")
    parser.add_argument("--mvg_pred_pos", type=str, default="full")

    parser.add_argument("--use_gnn", action="store_true", default=False)
    parser.add_argument("--embed", action="store_true", default=False)
    parser.add_argument("--nlayers_gnn", type=int, default=3)
    parser.add_argument("--n_message", type=int, default=1)
    parser.add_argument("--n_edge_layers", type=int, default=1)
    parser.add_argument("--use_myTransformer", action="store_true", default=False)
    parser.add_argument("--use_fast_transformer", action="store_true", default=False)
    parser.add_argument("--transformer_type", type=str, default="torch")
    parser.add_argument("--fast_transformer_backend", type=str, default="flash")

    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--pert_pad_id", type=int, default=2)
    
    parser.add_argument("--pool_size", type=int, default=300)
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--mvg_decoder_style", type=str, default="continuous")
    parser.add_argument("--focal_alpha", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--train_stage", type=str, default=None)
    parser.add_argument("--negative_ratio", type=float, default=0.9)
    parser.add_argument("--five_class", action="store_true", default=False)
    parser.add_argument("--test_mode", type=str, default="dataset")
    parser.add_argument("--text_emb_path", default=None)
    parser.add_argument("--edge_path", default=None)
    
    args = parser.parse_args()
    return args

def predict(
    model: PertModel, 
    dataset: PertDataset,
    pert_list: List[str], 
    pool_size: Optional[int] = None,
    args: Any = None,
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = dataset.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    gene_list = dataset.gene_names
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )
    gene_ids = dataset.gene_vocab_id
    node_map = {x: it for it, x in enumerate(gene_list)}

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=args.batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, node_map, gene_ids=gene_ids
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred

def plot_perturbation(
    model: nn.Module, 
    dataset: PertDataset,
    query: str, 
    save_file: str = None, 
    pool_size: int = None,
    args: Any = None,
):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = dataset.adata
    gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
    if query.split("+")[1] == "ctrl":
        pred = predict(model, dataset, [[query.split("+")[0]]], pool_size=pool_size, args=args)
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(model, dataset, [query.split("+")], pool_size=pool_size, args=args)
        pred = pred["_".join(query.split("+"))][de_idx]
    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    plt.figure(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax = plt.gca()
    ax.xaxis.set_ticklabels(genes, rotation=45)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        plt.savefig(save_file, bbox_inches="tight", transparent=False)
    # plt.show()

def eval_perturb(
    model: nn.Module, dataset: PertDataset, args: Any = None,
) -> Dict:
    """
    Run model in inference mode using a given dataset
    """

    model.eval()
    data_len = dataset.data_len
    device = next(model.parameters()).device
    model.to(device)
    dtype = model.model.dtype
    bsz = args.batch_size
    logger.info(f"[*]model device: {device}")
    logger.info(f"[*]model dtype: {dtype}")
    logger.info(f"[*]batch size: {bsz}")

    pert_cat = []
    preds = []
    truths = []
    preds_de = []
    truths_de = []
    values = []
    values_de = []
    results = {}
    logvar = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
        for i_batch in trange(0, data_len, bsz):
            data_batch = [
                dataset.__getitem__(idx) 
                for idx in range(i_batch, i_batch+bsz) if idx < data_len
            ]
            batch = dataset.collater(data_batch)
            batch = {
                k: v.to(device) if type(v)==torch.Tensor else v 
                for k, v in batch.items()
            }

            output_dict = model(batch)
            pred = output_dict["mlm_output"]
            target = batch['target']
            value = batch['values']
            pert_cat.extend(batch['pert_name'])
            preds.append(pred.detach().cpu())
            truths.append(target.detach().cpu())
            values.append(value.detach().cpu())
            for itr, de_idx in enumerate(batch['de_idx']):
                preds_de.append(pred[itr, de_idx].detach().cpu())
                truths_de.append(target[itr, de_idx].detach().cpu())
                values_de.append(value[itr, de_idx].detach().cpu())
    results["value"] = torch.cat(values, dim=0).numpy().astype(np.float)
    results["value_de"] = torch.stack(values_de, dim=0).numpy().astype(np.float)
    results["pert_cat"] = np.array(pert_cat)
    results["pred"] = torch.cat(preds, dim=0).numpy().astype(np.float)
    results["truth"] = torch.cat(truths, dim=0).numpy().astype(np.float)
    results["pred_de"] = torch.stack(preds_de, dim=0).numpy().astype(np.float)
    results["truth_de"] = torch.stack(truths_de, dim=0).numpy().astype(np.float)
    return results

def basic_metric(our_res):
    our_pred, our_pred_de = our_res['pred'], our_res['pred_de']
    truth, truth_de =  our_res['truth'], our_res['truth_de']
    input, input_de = our_res['value'], our_res['value_de']
    truth = truth.reshape(-1)
    truth_de = truth_de.reshape(-1)
    input = input.reshape(-1)
    input_de = input_de.reshape(-1)
    our_pred = our_pred.reshape(-1)
    our_pred_de = our_pred_de.reshape(-1)
    our_mse, our_mse_de = mse(our_pred, truth), mse(our_pred_de, truth_de)
    our_corr, our_corr_de = pearsonr(our_pred, truth)[0], pearsonr(our_pred_de, truth_de)[0]
    our_delta_corr, our_delta_corr_de = pearsonr(our_pred-input, truth-input)[0], pearsonr(our_pred_de-input_de, truth_de-input_de)[0]
    print(f"Metric          mse               mse_de               corr                corr_de              corr_delta            corr_delta_de")
    print(f"ours :  {our_mse}  {our_mse_de}  {our_corr}  {our_corr_de}  {our_delta_corr}    {our_delta_corr_de}")

def analyse_correct_direction(results, adata):
    truths = results['truth']
    inputs = results['value']
    preds = results['pred']
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    # geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
    # num_genes = truths.shape[0] * truths.shape[1]

    pert_metric = {
        'frac_correct_direction_all': 0,
        'frac_correct_direction_20': 0,
        'frac_correct_direction_50': 0,
        'frac_correct_direction_100': 0,
        'frac_correct_direction_200': 0,
    }
    for pert in np.unique(results['pert_cat']):
        de_idx = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:20]]
        de_idx_200 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:200]]
        de_idx_100 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:100]]
        de_idx_50 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:50]]

        pert_idx = np.where(results['pert_cat'] == pert)[0]  
 
        direc_change = np.abs(np.sign(preds[pert_idx] - inputs[pert_idx]) - np.sign(truths[pert_idx] - inputs[pert_idx]))  
        num_correct_direction = len(np.where(direc_change == 0)[0])/truths.shape[1]
        pert_metric['frac_correct_direction_all'] += num_correct_direction

        de_idx_map = {20: de_idx,
                50: de_idx_50,
                100: de_idx_100,
                200: de_idx_200
                }
        
        for val in [20, 50, 100, 200]:
            direc_change = np.abs(np.sign(preds[pert_idx][:,de_idx_map[val]] - inputs[pert_idx][:,de_idx_map[val]]) - \
                np.sign(truths[pert_idx][:,de_idx_map[val]] - inputs[pert_idx][:,de_idx_map[val]]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/val
            # pert_metric['frac_correct_direction_' + str(val)] = frac_correct_direction
            pert_metric['frac_correct_direction_' + str(val)] += frac_correct_direction

    for val in [20, 50, 100, 200]:
        key = 'frac_correct_direction_' + str(val)
        pert_metric[key] = pert_metric[key]/truths.shape[0]
        print(f"{key}: {pert_metric[key]}, {pert_metric[key]*val*truths.shape[0]}")
    pert_metric['frac_correct_direction_all'] /= truths.shape[0]
    print(f"'frac_correct_direction_all': {pert_metric['frac_correct_direction_all']}, \
        {pert_metric['frac_correct_direction_all']*truths.shape[1]*truths.shape[0]}")
  

if __name__ == "__main__":

    args = get_arguments()
    logger.info(f"[***]Args: {args}")

    if args.debug_mode:
        args.best_model=None
        args.nlayers = 1
        args.nhead = 4
        args.d_hid = args.d_model = 64
        args.batch_size = 2
        args.transformer_type = 'torch'

    adata_path = os.path.join(args.data_path, 'pert_adata.pkl')
    with open(adata_path, 'rb') as f:
        adata = pickle.load(f)
    print(f"adata:  {adata}")

    ## prepare model
    model = PertModel(args)
    logger.info(f"[***]Model architecture:")
    logger.info(f"{model}:")
    # args.best_model=None
    if args.best_model:
        pretrained_dict = torch.load(args.best_model)['model']
        model.load_state_dict(pretrained_dict)
        logger.info(f"[*]Loading all model params from {args.best_model}")
    else:
        logger.info("WARNING: No tested model!")
        # raise ValueError("args.best_model should not be None!")
    model = model.float()
    model.cuda()

    ## prepare data
    dataset = PertDataset(
        data_path = args.data_path,
        vocab_path = args.vocab_path,
        use_gnn = args.use_graph,
        split = "test",
        mode = "test",
        shuffle = args.shuffle,
        use_embed=args.embed,
        text_emb_path = args.text_emb_path,
        edge_path=args.edge_path
    )
    dataset.adata = adata
    logger.info(f"[***]Successfully load dataset!")

    ## Plot pert 
    # logger.info(f"[**]Test: predict one given pert")
    # perts_to_plot = ["SAMD1+ZBTB1"]
    # for p in perts_to_plot:
    #     plot_perturbation(
    #         model, dataset, p, 
    #         pool_size=args.pool_size, 
    #         save_file=f"{args.save_dir}/{p}.png", 
    #         args = args,
    #     )

    results_path = f"{args.save_dir}/results.pickle"
    # if os.path.exists(results_path):
    #     logger.info(f"Path already exists, loading from file: {results_path}")
    #     with open(results_path, 'rb') as f:
    #         results = pickle.load(f)
    # else:
    # Testing begin
    logger.info(f"[**]Test in test dataset!")
    results = eval_perturb(model, dataset, args)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    metric_gears_path = f"{args.save_dir}/test_metrics.pickle"
    if os.path.exists(metric_gears_path):
        logger.info(f"Path already exists, loading from file: {metric_gears_path}")
        with open(metric_gears_path, 'r') as f:
            test_metrics = json.load(f)
        with open(f"{args.save_dir}/test_pert_res.json", 'r') as f:
            test_pert_res = json.load(f)
    else:
        test_metrics, test_pert_res = compute_metrics(results)
        with open(f"{args.save_dir}/test_metrics.json", "w") as f:
            json.dump(test_metrics, f)
        with open(f"{args.save_dir}/test_pert_res.json", "w") as f:
            json.dump(test_pert_res, f)
    logger.info(test_metrics)
    logger.info(test_pert_res)


    deeper_metrics_path = f"{args.save_dir}/deeper_results.pickle"
    if os.path.exists(deeper_metrics_path):
        logger.info(f"Path already exists, loading from file: {deeper_metrics_path}")
        with open(deeper_metrics_path, 'rb') as f:
            deeper_metrics = pickle.load(f)
    else:
        deeper_metrics = deeper_analysis(adata, results)
        with open(deeper_metrics_path, 'wb') as f:
            pickle.dump(deeper_metrics, f)
    # logger.info(deeper_metrics)
    
    basic_metric(results)
    analyse_correct_direction(results, adata)

    logger.info(f"[**]Testing finish!")
    # import pdb;pdb.set_trace()
