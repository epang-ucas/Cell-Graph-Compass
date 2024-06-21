import argparse, copy, os
from typing import List, Tuple, Dict, Union, Optional
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

# from sklearn.metrics import confusion_matrix
import scib
from anndata import AnnData
import anndata as ad
import scanpy as sc
import pickle

# from models.clustering_model import ClusteringModel

import torch
import torch.nn as nn

from data.single_cell_dataset import ScDataset
from models.single_cell_model import ScModel

import time
print(time.ctime())

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
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed=embed,
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        print(f"{notes}")

    print(f"{results}")

    result_dict = results[0].to_dict()
    print(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
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

    return result_dict

import torchmetrics
# from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import trange
def evaluate(
    model,
    dataset,
    data_total,
    args,
    model_path,
    draw_graph=True,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    check_num = model_path.split("/")[-1].split('.')[0].split('_')[-1]
    pretrained_dict = torch.load(model_path)['model']
    try:
        model.load_state_dict(pretrained_dict)
        print(f"[*]Loading all model params from {model_path}")
    except:
        model_dict = model.state_dict()
        load_dict = dict()
        for key in model_dict.keys():
            if key not in pretrained_dict:
                print(f"[*]model param: {key} not in pretrain model")
            else:
                if key.split(".")[1] == "dsbn":
                    print(f"[*]model param: {key} will be trained from scatch")
                    continue
                if pretrained_dict[key].shape != model_dict[key].shape:
                    print(f"[*]model param: {key} shape is not consistent with pretrain model")
                    print(f"[*]model shape: {model_dict[key].shape}, pretrain model shape: {pretrained_dict[key].shape}")
                else:
                    load_dict[key] = pretrained_dict[key]
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)
        print("[**]Load model successfully!")
    model = model.float()
    model.cuda()
    print("Start evaluate")
    model.eval()
    num_types = model.n_cls
    device = next(model.parameters()).device
    metric_collection = torchmetrics.MetricCollection({ 
        'acc': torchmetrics.Accuracy(task="multiclass", num_classes=num_types, average='micro'), 
        'prec': torchmetrics.Precision(task="multiclass", num_classes=num_types, average='none'),
        'rec': torchmetrics.Recall(task="multiclass", num_classes=num_types, average='none'),
        'F1score': torchmetrics.F1Score(task="multiclass", num_classes=num_types, average='none')
    }) 
    metric_collection = metric_collection.to(device)
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_types).to(device)
    confmat_norm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_types,normalize='true').to(device)
    total_cls = 0.0
    data_len = dataset.data_len
    batch_size = args.batch_size
    print("Start get item",data_len,batch_size)
    output_cls_all = []
    labels = []
    print("Start range")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
        batch_data_={}
        for i in trange(0, data_len, batch_size):
            data_batch = data_total[i : i + batch_size]
            batch_data = dataset.collater(data_batch)
            for k, v in batch_data.items():
                if v!=None:
                    batch_data_[k]=v.to(device)
                else:
                    batch_data_[k]=v
            output_dict = model(batch_data_)
            output_cls = output_dict["cls_output"]
            output_cls_all.append(output_dict["cls_output"].detach().cpu())
            celltype_labels = batch_data_['celltype'].long()
            labels.append(celltype_labels.detach().cpu())
            loss_cls = nn.CrossEntropyLoss()(output_cls, celltype_labels)
            metric_collection.update(output_cls, celltype_labels)
            confmat.update(output_cls, celltype_labels)
            confmat_norm.update(output_cls, celltype_labels)
    total_cls += loss_cls.item()
    total_metrics  = metric_collection.compute()
    metric_collection.reset()
    cm = confmat.compute().cpu().numpy()
    cm_norm = confmat_norm.compute().cpu().numpy()
    output_cls_all = torch.concat(output_cls_all, dim=0).numpy()
    labels = torch.concat(labels, dim=0).numpy()
    print(f"total_cls {total_cls} ")
    print(f"total_metrics {total_metrics} ")
    print(f"total_metrics acc:{total_metrics['acc']} prec:{torch.mean(total_metrics['prec'])} rec:{torch.mean(total_metrics['rec'])} f1:{torch.mean(total_metrics['F1score'])}")
    print(f"valid/confmat: {cm}")
    # if draw_graph:
    #     with open(os.path.join(args.save_dir,f"result_{check_num}.pickle"),"wb") as f:
    #         pickle.dump((output_cls_all,labels,total_metrics,cm,cm_norm),f)

    #     plt.figure()
    #     ax = sn.heatmap(cm, cmap="YlGnBu")
    #     ax.set_xlabel("pred")
    #     ax.set_ylabel("truth")
    #     ax.set_title(f"acc:{total_metrics['acc']:.4f} prec:{torch.mean(total_metrics['prec']):.4f} rec:{torch.mean(total_metrics['rec']):.4f} f1:{torch.mean(total_metrics['F1score']):.4f}")
    #     plt.savefig(os.path.join(args.save_dir,f"ConfusionMatrix[{check_num}]_abs_.png"), dpi=1000)
    #     plt.show()
    #     plt.figure()    
    #     # ax = sn.heatmap(cm_norm, cmap="YlGnBu",annot=True,fmt='.2f',annot_kws={'size':4,'weight':'bold', 'color':'blue'})
    #     ax = sn.heatmap(cm_norm, cmap="YlGnBu",fmt='.2f')
    #     plt.savefig(os.path.join(args.save_dir, f"ConfusionMatrix[{check_num}]_norm_.png"), dpi=1000)
    #     plt.show()
    #     plt.figure()
    #     network_pal = sn.husl_palette(output_cls_all.shape[1], s=.45)
    #     sort_labels=np.argsort(labels)
    #     print(labels.shape,output_cls_all.shape)
    #     celltype = [network_pal[i] for i in labels[sort_labels]]
    #     output_cls_all=output_cls_all[sort_labels]
    #     sn.clustermap(data=output_cls_all,row_cluster=False,row_colors=celltype)
    #     plt.savefig(os.path.join(args.save_dir, f"HeatMap[{check_num}]_.png"), dpi=1000)
    #     plt.show()
    # print("End evaluate")
    return check_num,total_metrics['acc'],torch.mean(total_metrics['prec']),torch.mean(total_metrics['rec']),torch.mean(total_metrics['F1score']),model_path


if __name__ == "__main__":

    args = get_arguments()
    print(f"[**]args: {args}")

    model = ScModel(args)
    # prepare data
    dataset = ScDataset(
        data_path = args.data_path,
        vocab_path = args.vocab_path, 
        seq_max_len = args.seq_max_len,
        use_gnn = args.use_graph,
        edge_corr_thr = args.edge_corr_thr,
        split = "valid",
        mode = "valid",
        shuffle = args.shuffle,
        preprocess = args.data_preprocess,
        use_embed = args.embed,
        text_emb_path= args.text_emb_path,
    )

    print(f"[**]Successfully load dataset!")
    print(f"[*] The size of dataset is {dataset.__len__()}")
    print(f"[*] The size of gene vocab is {len(dataset.vocab)}")
    print(f"[*] The number of seq-batch is {dataset.batch_num}")
    print(f"[*] The set of celltype is {dataset.n_cls}")
    data_total = [dataset.__getitem__(i) for i in range(dataset.data_len)]
    print("Test for single checkpoint")
    # Testing begin
    print(f"[**]Testing begin!")
    # results = eval_testdata(model, dataset, args)
    evaluate(model,dataset,data_total,args,args.best_model)
    print(f"[**]Testing finish!")
    

   


        