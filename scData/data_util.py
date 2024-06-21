import torch

import pandas as pd
import scanpy as sc
import anndata as ad
import dgl

import numpy as np

from torchtext.vocab import Vocab
import pickle

from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from scgpt.tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor

class data_util(object):
    def __init__(self):
        pass
        
    @classmethod
    def save_graph(
        cls,
        edge_txt_path: str,
        graph_save_path: str
        ):
        """
        transform edge_txt into graph object.
        Args:
            edge_txt_path: the txt file path saving the edge 
                        with the format src\ttarget from line 1 to the end of file.
            graph_save_path: the pickle path to save the graph.
                        g: graph object.
                        tf_s: Series for gene_name:idx.
                        tf_s_inv: Series for idx:gene_name.
                        tf_gene_set: Set for gene_name.
        """
        tf_tar_pairs_set = set()
        with open(edge_txt_path,'r') as file:
            tf_tar_pairs = file.readlines()
        gene_dict = {}
        for i in range(1,len(tf_tar_pairs)):
            tf_gene, target_gene = tf_tar_pairs[i].strip().split('\t')
            if tf_gene==target_gene:
                print(tf_gene,target_gene)
            elif tf_gene.upper()==target_gene.upper():
                print(tf_gene,target_gene)
            tf_tar_pairs_set.add((tf_gene.upper(), target_gene.upper()))
            if gene_dict.get(tf_gene.upper())==None:
                gene_dict[tf_gene.upper()]=len(gene_dict)
            if gene_dict.get(target_gene.upper())==None:
                gene_dict[target_gene.upper()]=len(gene_dict)

        a=[]
        b=[]
        for (i,j) in tf_tar_pairs_set:   
            a.append(gene_dict[i])
            b.append(gene_dict[j])

        a_t,b_t = torch.tensor(a),torch.tensor(b)

        g = dgl.graph((a_t,b_t))
        tf_s=pd.Series(index=gene_dict.keys(),data=gene_dict.values())
        tf_s_inv=pd.Series(index=gene_dict.values(),data=gene_dict.keys())
        tf_gene_set=set(gene_dict.keys())

        with open(graph_save_path,'wb') as graph_f:
            pickle.dump((g,tf_s,tf_s_inv,tf_gene_set),graph_f)

    @classmethod
    def tokenize(
        cls,
        vocab,
        genes: np.ndarray,
        values: np.ndarray,
        include_zero_gene: bool = True,
        append_cls: bool = True,
        cls_token: bool = "<cls>",
        return_pt: bool = True,
    ):
        """
        Tokenize a sample of data. Returns tokenized values and gene names.

        Args:
            data (array-like): A sample of data, with shape (n_features,).
                n_features equals the number of all genes.
            gene_ids (array-like): A sample of gene ids, with shape (n_features,).
            return_pt (bool): Whether to return torch tensors of gene_ids and counts,
                default to True.

        Returns:
            gene_ids: padded gene_tokens.
            values: padded expresion values.
        """
        if len(genes) != len(values):
            raise ValueError(
                f"Number of features in data ({len(genes)}) does not match "
                f"number of gene_ids ({len(values)})."
            )
        gene_ids = np.array(vocab(genes.tolist()))
        cls_id = vocab[cls_token]
        if not include_zero_gene:
            idx = np.nonzero(values)
            values = values[idx]
            gene_ids = gene_ids[idx]
        if append_cls:
            gene_ids = np.append(cls_id,gene_ids)
            values = np.append(0,values)
            # gene_ids = np.insert(gene_ids, 0, cls_id)
            # values = np.insert(values, 0, 0)
        if return_pt:
            gene_ids = torch.from_numpy(gene_ids).long()
            values = torch.from_numpy(values)
        return gene_ids, values
    
    @classmethod
    def pad(
        cls,
        vocab,
        gene_ids,
        values,
        max_len: int,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        cls_appended: bool = True,
    ):
        """
        Pad a sample of data. Returns padded values and gene names.
        Args:
            vocab: the Vocab object map gene to vocab_id.
            gene_ids: the gene_id list.
            values: the binned gene expresion values list.
            max_len: the max length to crop.
            pad_token: special token for padding.
            pad_value: vocab_id for pad_token.
            cls_appended: whether appended cls token.
        Returns:
            gene_ids: padded gene_tokens.
            values: padded expresion values.
        """
        pad_id = vocab[pad_token]
        idx =  np.array(range(len(gene_ids)))
        if len(gene_ids) > max_len:
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.append(0, idx)
            gene_ids = gene_ids[idx]
            values = values[idx]
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full(
                        (max_len - len(values),), pad_value, dtype=values.dtype
                    ),
                ]
            )
            
        return gene_ids, values, idx
    
    @classmethod
    def sample(
        cls,
        arr: np.array,
        sample_length: int
        ):
        """
        Sample nonzero value with limited length where higher value has higher probability.
        Args:
            arr: expresion value array.
            sample_length: max sample length .
        Returns:
            res: sampled non_zero expresion value array
        """
        arr_nz = arr.nonzero()[0]
        if len(arr_nz)<=sample_length:
            return arr_nz
        arr_np = torch.from_numpy(arr.astype(np.float32))
        res = torch.multinomial(arr_np, num_samples=sample_length, replacement=False).numpy()
        return res

    @classmethod
    def sub_graph(
        cls,
        tf_s,
        tf_s_inv,
        g_tftg,
        tf_gene_set,
        data_gene,
        gene_series,
        return_attr=False,
        ):
        """
        Get sub_graph with given gene list
        Args:
            tf_s: gene->id Series object.
            tf_s_inv: id->gene Series object.
            g_tftg: graph object.
            tf_gene_set: gene set object.
            data_gene: gene_name list.
            gene_Series: gene->local rank.
            return_attr: whether return edge attr. True for chromosome graph attr.
        Returns:
            tftg_idx: the sub grpah edge list with local rank id in data_gene.
            tf_edge.edata['dis']: the attr for distance between two genes in the same chromosome.
        """
        node_l = tf_s[set(data_gene)&tf_gene_set].values
        tf_edge = dgl.node_subgraph(g_tftg,node_l)
        tf_node = tf_edge.ndata[dgl.NID]
        tftg = tf_edge.edges()
        r_0 = torch.gather(tf_node,0,tftg[0])
        r_1 = torch.gather(tf_node,0,tftg[1])
        i_0=gene_series[tf_s_inv[r_0.tolist()].values].values
        i_1=gene_series[tf_s_inv[r_1.tolist()].values].values
        tftg_idx = torch.from_numpy(np.stack((i_0,i_1)))
        if not return_attr:
            return tftg_idx
        else:
            return tftg_idx,tf_edge.edata['dis']


    @classmethod
    def savetoLibBatch(
        cls,
        env,
        lib,
        file_id,
        idx_train,
        celltype_set,
        gene_set,
        batch_set
    ):
        """
        Save global info to lmdb base for each file.
        Args:
            env: lmdb env object.
            lib: lmdb lib object.
            file_id: current file num.
            idx_train: current sample num.
            celltype_set: global cellltype_set.
            gene_set: global gene set.
            batch_set: global batch set.
        """
        lib.put(key = "file_num".encode(), value = pickle.dumps(file_id))
        lib.put(key = "sample_num".encode(),value = pickle.dumps(idx_train))
        lib.put(key = "celltype_num".encode(),value = pickle.dumps(len(celltype_set)))
        lib.put(key = "celltype_set".encode(),value = pickle.dumps(celltype_set))
        lib.put(key = "gene_num".encode(),value = pickle.dumps(len(gene_set)))
        lib.put(key = "gene_set".encode(),value = pickle.dumps(gene_set))
        lib.put(key = "batch_num".encode(),value = pickle.dumps(len(set(batch_set.values()))))
        lib.put(key = "batch_set".encode(),value = pickle.dumps(batch_set))
        lib.commit()
        env.close()


    @classmethod
    def graph_generate(
        cls,
        gene_list: np.ndarray,
        tftg_list: torch.Tensor,
        chromo_list: torch.Tensor,
        chromo_dis: torch.Tensor,
        corr_matrix: torch.Tensor,
        corr_thr: float = 0.6,
        cls_append: bool = True,
        max_dis: float=50.0,
        max_seq_len: int=1200,
    ):
        """
        Extract a subset from gene_list, tftg_list and corr_matrix,generate graph index and attr.
        Args:
            gene_list: array for gene.
            tftg_list: tftg_edge list.
            chromo_list: chromo_edge list.
            chromo_dis: the distance attr for chromosome edge.
            corr_matrix: corr for matrix.
            corr_thr: threshold for corr.
            cls_append: whether append cls.
            max_dis: max distance between genes.
        Returns:
            edge_index_res: edge index array.
            edge_attr_res: edge attr array,shape: [5, nedges].
            edge_attr_res[0, :] represents if this edge has tftg relation.
            edge_attr_res[1, :] represents if this edge has high corr.
            edge_attr_res[2, :] represents if this edge in same chromosome.
            edge_attr_res[3, :] represents this edge(gene pair)'s correlation strength.
            edge_attr_res[4, :] represents this edge(gene pair)'s distance in chromosome. 
                                map [-max_dis,max_dis] to [1,2*max_dis+1]
        """
        # import pdb;pdb.set_trace()
        nnode = len(gene_list)
        tftg_edge_idx=tftg_list[0]*max_seq_len+tftg_list[1]
        chromo_edge_idx=chromo_list[0]*max_seq_len+chromo_list[1]
        # tftg_tuple=list(zip(tftg_list[0].tolist(),tftg_list[1].tolist()))
        # chromo_tuple=list(zip(chromo_list[0].tolist(),chromo_list[1].tolist()))

        corr_high_list=torch.where(abs(corr_matrix)>corr_thr)
        # corr_tuple=list(zip(corr_high_list[0].tolist(),corr_high_list[1].tolist()))
        corr_edge_idx=corr_high_list[0]*max_seq_len+corr_high_list[1]

        
        all_edge=torch.cat([tftg_edge_idx,chromo_edge_idx,corr_edge_idx]).unique()
        nedges=len(all_edge)
        all_edge_s=pd.Series(index=all_edge.tolist(),data=range(nedges))

        edge_index_res=torch.stack([all_edge//max_seq_len,all_edge%max_seq_len],dim=0)
        edge_attr_res=torch.zeros((5,nedges))
        
        tftg_idx=all_edge_s[tftg_edge_idx.tolist()].values
        corr_idx=all_edge_s[corr_edge_idx.tolist()].values
        chromo_idx=all_edge_s[chromo_edge_idx.tolist()].values


        corr_all_edge=corr_matrix[edge_index_res[0],edge_index_res[1]]
        mapped_value=chromo_dis+max_dis+1.0

        edge_attr_res[0,tftg_idx]=1
        edge_attr_res[1,corr_idx]=1
        edge_attr_res[2,chromo_idx]=1
        edge_attr_res[3,:]=corr_all_edge
        edge_attr_res[4,chromo_idx]=mapped_value

        
        if cls_append:
            edge_index_res+=1
            edge_index_cls = torch.as_tensor([np.arange(1,nnode+1),[0]*nnode])
            edge_index_res = torch.cat([edge_index_res, edge_index_cls], dim=1)
            edge_attr_cls = torch.zeros((5,nnode)) 
            edge_attr_res = torch.cat([edge_attr_res, edge_attr_cls], dim=1)
        return edge_index_res, edge_attr_res
    
    @classmethod
    def getAdata(cls,adata_dir,reverse=False):
        """
        Read adata from h5ad or pickle file.
        Args:
           adata_dir: file path for adata.
           reverse: whether transpose data into cell*gene, sometimes raw data data is gene*cell.
        Return:
            adata/bdata: adata object.
        """
        if type(adata_dir)!=type(""):
            return adata_dir
        if adata_dir.split('.')[-1]=='h5ad':
            adata = sc.read(adata_dir,cache=True)
        elif adata_dir.split('.')[-1]=='pickle':
            with open(adata_dir,'rb') as f:
                adata = pickle.load(f)
        if not reverse:
            return adata
        else:
            bdata = ad.AnnData(adata.X.T,obs=adata.var,var=adata.obs)
        return bdata

    @classmethod
    def getCelltype(cls,adata_train_dir,adata_test_dir,key_celltype='Celltype',reverse=False):
        """
        Get celltype and update celltype id info into adata
        Args:
           adata_train_dir: file path for adata.
           adata_test_dir: empty str means only read data from adata_train_dir,
                           else read from both and concat them.
           key_celltype: key for celltype in adata.
           reverse: whether transpose data into cell*gene, used by getAdata method.
        Return:
            celltype_set: dict for celltype : id.
            adata: adata object with celltype id info.
        """
        if adata_test_dir!="":
            adata_test = cls.getAdata(adata_test_dir,reverse=reverse)
            adata_train = cls.getAdata(adata_train_dir,reverse=reverse)
            adata_test.obs["my_split"]="valid"
            adata_train.obs["my_split"]="train"
            adata = sc.AnnData.concatenate(adata_train,adata_test,batch_key="split")
        else:
            adata = cls.getAdata(adata_train_dir,reverse=reverse)
            adata.obs["my_split"]="all"
        celltype_id_labels = adata.obs[key_celltype].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        celltypeset = adata.obs[key_celltype].astype("category").cat.categories
        celltype_set = dict(zip(celltypeset,range(len(celltypeset))))
        return celltype_set,adata
        
    @classmethod
    def filter_adata(cls,adata_dir,vocab_dir,key_celltype="Celltype",key_batch='batch',reverse=False):
        """
        Filter adata with vocab.
        Args:
            adata_dir: str means read according file path,
                    else it is an adata object.
            vocab_dir: json file for vocab(id:gene name).
            key_celltype: key for celltype.
            key_batch: key for batch.
            reverse: whether transpose data into cell*gene, used by getAdata method.
        Returns:
            adata: filterd adata object.
            vocab: geneVocab object.
        """
        if type(adata_dir)==type(""):
            adata = cls.getAdata(adata_dir,reverse=reverse)
        else:
            adata = adata_dir
        vocab = GeneVocab.from_file(vocab_dir)

        if 'gene_name' in adata.var.columns:
            adata.var.index = adata.var["gene_name"]
            adata.var_names_make_unique()
            adata.var["gene_name"] = adata.var.index.str.upper()

        if "gene_name" not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.str.upper()
        adata.var["id_in_vocab"] = [ 1 if gene in vocab else -1 for gene in adata.var["gene_name"] ]

        if key_batch not in adata.obs.columns:
            adata.obs['batch']=0
        else:
            adata.obs['batch']=adata.obs[key_batch]

        adata.obs['batch']=adata.obs['batch'].astype("category")
        if "cell_type" not in adata.obs.columns:
            adata.obs["cell_type"]=adata.obs[key_celltype]    
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        return adata,vocab
    
    @classmethod
    def getCorr(cls,adata,gene_set):
        """
        Get corrcoef by gene expression value matrix.
        Args:
            adata: adata object.
            gene_set: dict for gene(gene name:id)
        Return:
            X: gene expresion value.
            corr: corr matrix.
            gene_array: array for gene_name.
        """
        X = adata.layers["X_binned"]
        gene_name =  adata.var["gene_name"]
        gene_list = gene_name.to_list()
        gene_set = cls.merge(gene_set,gene_list)
        X_torch = torch.from_numpy(X)
        corr = torch.corrcoef(X_torch.T)
        row, col = np.diag_indices_from(corr)
        corr[row,col] = 0
        print(torch.isnan(corr).sum())
        corr = torch.where(torch.isnan(corr), torch.full_like(corr, 0), corr)        
        gene_array = np.array(gene_list)
        return X,corr,gene_array
    
    @classmethod
    def merge(cls,gene_set,gene_list):
        """
        Merge gene list into gene dict.
        Args:
            gene_set: dict for gene(gene name:id)
            gene_list: list for gene name.
        Return:
            gene_set: merged gene dict.
        """
        for i in gene_list:
            if gene_set.get(i)==None:
                gene_set[i]=len(gene_set)
        return gene_set

def scPipeline(adata,ishvg=True,data_is_raw=True,n_hvg=1200,filter_gene_by_counts=3,n_bins = 51):
    """
    Pipeline for process adata.
    Args:
        adata: adata obeject.
        ishvg: whether do hvg operation.
        data_is_raw: whether data is raw, False means it has done log1p operation.
        n_hvg: hvg length, used when ishvg is True.
        filter_gene_by_counts: filter gene counts in cells < the set num.
    Return:
        adata: processed adata.
    """
    preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=True,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg if ishvg else None,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key="batch")
    return adata
