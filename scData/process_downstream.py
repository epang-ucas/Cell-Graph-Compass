import pickle
import os
import lmdb
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_util import data_util
from data_util import scPipeline
from scgpt.tokenizer import GeneVocab


class Preprocess:

    def __init__(
        self,
        adata_path,
        train_save_dir,

        test_save_dir="",
        is_hvg=True,
        n_hvg=1200,
        use_gnn=True,
        data_is_raw=True,
        has_filter=False,
        include_zero=True,

        vocab_path="./bioFeature_embs/vocab.json",
        tf_tg_graph_path="./bioFeature_embs/tf_tg_graph",
        chromo_graph_path="./bioFeature_embs/chromo_graph",
    ):
        """
        Init for data preprocess class.
        Args:
            adata_path: file path for adata.
            train_save_dir: save dir for train split data.
            test_save_dir: save dir for test split data, empty means no test split.
            is_hvg: whether select hvgs.
            n_hvg: the number of hvgs.
            use_gnn: whether contains graph information.
            data_is_raw: whether data has been log1p.
            has_filter: whether data has benn processed by scPipeline.
            include_zero: whether contains genes with zero expresion value. 
                        If true,all cells will contain the same gene list.
            vocab_path: path for vocab file.
            tf_tg_graph_path: path for tf_tg graph file.
            chromo_graph_path: path for chromo_graph file.
        """
        self.adata_path = adata_path
        self.train_save_dir = train_save_dir
        self.test_save_dir = test_save_dir

        self.is_hvg = is_hvg
        self.n_hvg = n_hvg
        self.seq_max_len = n_hvg
        self.use_gnn = use_gnn
        self.data_is_raw = data_is_raw
        self.has_filter = has_filter
        self.include_zero=include_zero

        self.vocab_path = vocab_path
        self.tf_tg_graph_path = tf_tg_graph_path
        self.chromo_graph_path = chromo_graph_path
        
        
    def make_lmdb_data(self,adata,lmdb_dir,lmdb_test_dir,vocab,celltype_set):
        """
        Save lmdb data.
        Args:
            adata: adata object.
            lmdb_dir: save dir for train split data.
            lmdb_test_dir: save dir for test split data, empty means no test split.
            vocab: dict for genes.
            celltype_set: dict for celltypes.
        """
        adata = adata[~adata.obs['cell_type'].isnull(),:]
        if not self.has_filter:
            adata = scPipeline(adata,ishvg=self.is_hvg,n_hvg=self.n_hvg,data_is_raw=self.data_is_raw,filter_gene_by_counts=0)
        gene_set = dict()
        X,corr,gene_array = data_util.getCorr(adata,gene_set)
        celltype = adata.obs["cell_type"]
        batch = adata.obs["batch"]
        batchset = adata.obs['batch'].cat.categories
        basedir=os.path.abspath(os.path.join(lmdb_dir,os.pardir))
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        if not os.path.exists(lmdb_dir):
            os.mkdir(lmdb_dir)
        batch_set = dict(zip(batchset,range(len(batchset))))
        env = lmdb.open(lmdb_dir, map_size=int(1099511627776*5))
        lib = env.begin(write=True)
        if lmdb_test_dir=="":
            self.enumerateAdata(gene_array,corr,X,vocab,
                        celltype_set,gene_set,batch_set,batch,celltype,
                        env,lib)
        else:
            (
                train_X,
                test_X,
                train_celltype,
                test_celltype,
                train_batch,
                test_batch,
            ) = train_test_split(
                X, celltype, batch, test_size=0.1, shuffle=True,random_state=42
            )
            env_test = lmdb.open(lmdb_test_dir, map_size=int(1099511627776*5))
            lib_test = env_test.begin(write=True)
            self.enumerateAdata(gene_array,corr,train_X,vocab,
                        celltype_set,gene_set,batch_set,train_batch,train_celltype,
                        env,lib)
            self.enumerateAdata(gene_array,corr,test_X,vocab,
                        celltype_set,gene_set,batch_set,test_batch,test_celltype,
                        env_test,lib_test)
       
    def enumerateAdata(self,gene_array,corr,X,vocab,
                        celltype_set,gene_set,batch_set,batch,celltype,
                        env,lib,idx_train=0,file_id=0):
        """
        Save lmdb data for one adata file.
        Args:
            gene_array: array for genes in the adata file.
            corr: corr matrix for all gene pairs.
            X: the matrix of gene expression value.
            vocab: dict for genes.
            celltype_set: the global dict for celltypes.
            gene_set: dict for aray of genes
            batch_set: the global dict for batch.
            batch: the batch list of all cells in the adata.
            celltype: the celltype list of all cells in the adata.
            env: lmdb env object.
            lib: lmdb lib object.
            idx_train: the global id for current data.
            file_id: id for adata files.
        """
        with open(self.tf_tg_graph_path,'rb') as graph_f:
            g_tf_tg,tf_s,tf_s_inv,tf_gene_set = pickle.load(graph_f)
        with open(self.chromo_graph_path,'rb') as graph_f:
            g_chromo,chromo_s,chromo_s_inv,chromo_gene_set = pickle.load(graph_f)

        if self.include_zero:
            if self.use_gnn:
                gene_series = pd.Series(index = gene_array,data = range(len(gene_array)))
                tftg_idx = data_util.sub_graph(tf_s,tf_s_inv,g_tf_tg,tf_gene_set,gene_array,gene_series)
                chromo_idx,chromo_dis=data_util.sub_graph(chromo_s,chromo_s_inv,g_chromo,
                                        chromo_gene_set,gene_array,gene_series,return_attr=True)
                corr_edge = corr
                index_edge, attr_edge = data_util.graph_generate(
                    gene_array,  # array (ngene,)
                    tftg_idx, # torch.Size([2, nedge1])
                    chromo_idx, # torch.Size([2, nedge2])
                    chromo_dis, # torch.Size([1,nedge2])
                    corr_edge,  # torch.Size([ngene, ngene])
                    corr_thr = 0.6,
                    cls_append = True,
                    max_dis=50.0,
                    max_seq_len=self.seq_max_len,
                )
                index_edge=index_edge.to(torch.int16)
                type_attr=attr_edge[0:3,:].to(torch.int8)
                corr_attr=attr_edge[3,:].to(torch.float16)
                chromo_attr=attr_edge[4,:].to(torch.int8)
            genes_b = gene_array
        for id,data in enumerate(X):
            if not self.include_zero:
                id_nonzero = data_util.sample(data,self.seq_max_len)
                data_gene = gene_array[id_nonzero]
                if self.use_gnn:
                    gene_series = pd.Series(index = data_gene,data = range(len(data_gene))) 
                    tftg_idx = data_util.sub_graph(tf_s,tf_s_inv,g_tf_tg,tf_gene_set,data_gene,gene_series)
                    chromo_idx,chromo_dis=data_util.sub_graph(chromo_s,chromo_s_inv,g_chromo,
                                        chromo_gene_set,data_gene,gene_series,return_attr=True)
                    id_nonzero_t = torch.from_numpy(id_nonzero)
                    x, y = torch.meshgrid(id_nonzero_t, id_nonzero_t)
                    corr_edge = corr[x,y]
                genes_b = data_gene
                values = data[id_nonzero]
            else:
                genes_b = gene_array
                values = data
            gene_ids, values = data_util.tokenize(
                vocab, genes_b, values, include_zero_gene=True, 
                    append_cls = True, cls_token = "<cls>", 
                    return_pt = True,
            )
            genes, values, _ = data_util.pad(
                vocab, gene_ids, values, max_len = self.seq_max_len+1,
                pad_token = "<pad>", pad_value = -2, 
                cls_appended = True,
            )
            values = values.to(torch.float)
            if self.include_zero and self.use_gnn:
                edge_index=index_edge
                edge_attr_type,edge_attr_corr,edge_attr_chromo=type_attr,corr_attr,chromo_attr
            elif self.include_zero==False and self.use_gnn:
                edge_index, edge_attr = data_util.graph_generate(
                    genes_b,  # array (ngene,)
                    tftg_idx, # torch.Size([2, nedge1])
                    chromo_idx, # torch.Size([2, nedge2])
                    chromo_dis, # torch.Size([1,nedge2])
                    corr_edge,  # torch.Size([ngene, ngene])
                    corr_thr = 0.6,
                    cls_append = True,
                    max_dis=50.0,
                    max_seq_len=self.seq_max_len,
                )
                edge_index=edge_index.to(torch.int16)
                edge_attr_type=edge_attr[0:3,:].to(torch.int8)
                edge_attr_corr=edge_attr[3,:].to(torch.float16)
                edge_attr_chromo=edge_attr[4,:].to(torch.int8)
            else:
                edge_index, edge_attr = None,None
                edge_attr_type,edge_attr_corr,edge_attr_chromo=None,None,None
            genes=genes.to(torch.int16)
            values=values.to(torch.int8)
            feature_dict = {
                'gene_list': genes, 
                'values': values, 
                'batch_id': batch_set[batch[id]], 
                'celltype': celltype_set[celltype[id]],
                'edge_index': edge_index, 
                'edge_attr_type': edge_attr_type,
                'edge_attr_corr': edge_attr_corr,
                'edge_attr_chromo':edge_attr_chromo,
                "idx":  idx_train,
            }
            key = f'{idx_train}'
            dict_serize = pickle.dumps(feature_dict)
            lib.put(key = key.encode(), value = dict_serize)
            idx_train+=1
            if id%1000==0:
                print(id)
        data_util.savetoLibBatch(env,lib,file_id,idx_train,celltype_set,gene_set,batch_set)



    def process_file(self,key_celltype="cell_type"):
        """
        Save lmdb data for no test split.
        Args:
            key_celltype: key for celltype in the file.
        """
        celltype_set,adata = data_util.getCelltype(self.adata_path,adata_test_dir="",key_celltype=key_celltype)
        adata,vocab = data_util.filter_adata(adata,self.vocab_path,key_celltype=key_celltype)
        
        self.make_lmdb_data(adata,lmdb_dir=self.train_save_dir,lmdb_test_dir="",vocab=vocab,celltype_set=celltype_set)


    
    def process_file_into_train_test(self,key_celltype="cell_type"):
        """
        Save lmdb data for train and test splits.
        Args:
            key_celltype: key for celltype in the file.
        """
        celltype_set,adata = data_util.getCelltype(self.adata_path,adata_test_dir="",key_celltype=key_celltype)
        adata,vocab = data_util.filter_adata(adata,self.vocab_path,key_celltype=key_celltype)
        self.make_lmdb_data(adata,lmdb_dir=self.train_save_dir,lmdb_test_dir=self.test_save_dir,vocab=vocab,celltype_set=celltype_set)
    
    def generate_edge_adata(self,corr,gene_array,cls_append):
        """
        Generate edges for all cells with the same gene list.
        Args:
            corr: corr matrix.
            gene_array: array for gene list in the cell.
            cls_append: whether append cls token.
        Return:
            index_edge: the union of edges for all attr.
            type_attr: edges for tf_tg attr.
            corr_attr: edges for corr attr.
            chromo_attr: edges for chromo attr.

        """
        assert self.use_gnn,"use_gnn is false thus no edge will be generated"
        assert self.include_zero,"include_zero is false thus genes are different among samples"
        assert len(gene_array)==self.n_hvg,"length of gene_list mismatch seq_max_len"
        with open(self.tf_tg_graph_path,'rb') as graph_f:
            g_tf_tg,tf_s,tf_s_inv,tf_gene_set = pickle.load(graph_f)
        with open(self.chromo_graph_path,'rb') as graph_f:
            g_chromo,chromo_s,chromo_s_inv,chromo_gene_set = pickle.load(graph_f)
        gene_series = pd.Series(index = gene_array,data = range(len(gene_array))) 
        tftg_idx = data_util.sub_graph(tf_s,tf_s_inv,g_tf_tg,tf_gene_set,gene_array,gene_series)
        chromo_idx,chromo_dis=data_util.sub_graph(chromo_s,chromo_s_inv,g_chromo,
                                        chromo_gene_set,gene_array,gene_series,return_attr=True)
        index_edge, attr_edge = data_util.graph_generate(
            gene_array,  # array (ngene,)
            tftg_idx, # torch.Size([2, nedge1])
            chromo_idx, # torch.Size([2, nedge2])
            chromo_dis, # torch.Size([1,nedge2])
            corr,  # torch.Size([ngene, ngene])
            corr_thr = 0.6,
            cls_append = cls_append,
            max_dis=50.0,
            max_seq_len=self.seq_max_len,
        )
        index_edge=index_edge.to(torch.int16)
        type_attr=attr_edge[0:3,:].to(torch.int8)
        corr_attr=attr_edge[3,:].to(torch.float16)
        chromo_attr=attr_edge[4,:].to(torch.int8)
        return index_edge,type_attr,corr_attr,chromo_attr

    def generate_edge_sample(self,corr,gene_array,bin_values,idx,batch_id,celltype_id,cls_append=True):
        """
         Generate edges for one cell.
        Args:
            corr: corr matrix.
            gene_array: array for gene list in the cell.
            bin_values: list for gene expression values.
            idx: global id for the data.
            batch_id: global batch id for the data.
            celltype_id: global celltype id for the data.
            cls_append: whether append cls token.
        Return:
            feature_dit: dict for preprocessed features.
        """
        assert self.use_gnn,"use_gnn is false thus no edge will be generated"
        # assert len(gene_array)==self.n_hvg,"length of gene_list mismatch seq_max_len"
        with open(self.tf_tg_graph_path,'rb') as graph_f:
            g_tf_tg,tf_s,tf_s_inv,tf_gene_set = pickle.load(graph_f)
        with open(self.chromo_graph_path,'rb') as graph_f:
            g_chromo,chromo_s,chromo_s_inv,chromo_gene_set = pickle.load(graph_f)
        vocab = GeneVocab.from_file(self.vocab_path)

        if not self.include_zero:
            id_nonzero = data_util.sample(bin_values,self.seq_max_len)
            gene_array = gene_array[id_nonzero]
            id_nonzero_t = torch.from_numpy(id_nonzero)
            x, y = torch.meshgrid(id_nonzero_t, id_nonzero_t)
            corr = corr[x,y]
            bin_values = bin_values[id_nonzero]

        gene_series = pd.Series(index = gene_array,data = range(len(gene_array))) 
        tftg_idx = data_util.sub_graph(tf_s,tf_s_inv,g_tf_tg,tf_gene_set,gene_array,gene_series)
        chromo_idx,chromo_dis=data_util.sub_graph(chromo_s,chromo_s_inv,g_chromo,
                                        chromo_gene_set,gene_array,gene_series,return_attr=True)

        index_edge, attr_edge = data_util.graph_generate(
            gene_array,  # array (ngene,)
            tftg_idx, # torch.Size([2, nedge1])
            chromo_idx, # torch.Size([2, nedge2])
            chromo_dis, # torch.Size([1,nedge2])
            corr,  # torch.Size([ngene, ngene])
            corr_thr = 0.6,
            cls_append = cls_append,
            max_dis=50.0,
            max_seq_len=self.seq_max_len,
        )

        index_edge=index_edge.to(torch.int16)
        type_attr=attr_edge[0:3,:].to(torch.int8)
        corr_attr=attr_edge[3,:].to(torch.float16)
        chromo_attr=attr_edge[4,:].to(torch.int8)
        genes_b = gene_array
        values = bin_values

        gene_ids, values = data_util.tokenize(
            vocab, genes_b, values, include_zero_gene=True, 
            append_cls = True, cls_token = "<cls>", 
            return_pt = True,
        )

        genes, values, _ = data_util.pad(
            vocab, gene_ids, values, max_len = self.seq_max_len+1,
            pad_token = "<pad>", pad_value = -2, 
            cls_appended = cls_append,
        )

        feature_dict = {
            'gene_list': genes, 
            'values': values, 
            'batch_id': batch_id, 
            'celltype': celltype_id,
            'edge_index': index_edge, 
            'edge_attr_type': type_attr,
            'edge_attr_corr': corr_attr,
            'edge_attr_chromo':chromo_attr,
            "idx":  idx,
        }
        return feature_dict

if __name__=="__main__":

   
    ###  PBMC 
    # adata_train_dir = "./train.pickle"
    # adata_test_dir = "./test.pickle"
    # celltype_set,adata = data_util.getCelltype(adata_train_dir,adata_test_dir,key_celltype='celltype')
    # pbmc_all=Preprocess(
    #     adata_path=adata,
    #     train_save_dir="./pbmc_lmdb/all",
    #     test_save_dir="",
    #     n_hvg=1200,
    # )
    # pbmc_all.process_file(key_celltype='celltype')

    # celltype_set,adata = data_util.getCelltype(adata_train_dir,adata_test_dir,key_celltype='celltype')
    # pbmc_all=Preprocess(
    #     adata_path=adata,
    #     train_save_dir="./pbmc_lmdb/train",
    #     test_save_dir="./pbmc_lmdb/test",
    #     n_hvg=1200,
    # )
    # pbmc_all.process_file_into_train_test(key_celltype='celltype')


    ## MS
    adata_train_dir = "./ms_train.pickle"
    adata_test_dir = "./ms_test.pickle"
    celltype_set,adata = data_util.getCelltype(adata_train_dir,adata_test_dir,key_celltype='cell_type')
    ms_train=Preprocess(
        adata_path=adata[adata.obs["my_split"]=="train"],
        train_save_dir="/train14/superbrain/zlhu12/data/Download/downstream/MS/latest_3_dim/train",
        test_save_dir="",
        n_hvg=3000,
        data_is_raw=False,
    )
    ms_train.process_file()

    celltype_set,adata = data_util.getCelltype(adata_train_dir,adata_test_dir,key_celltype='cell_type')
    ms_valid=Preprocess(
        adata_path=adata[adata.obs["my_split"]=="valid"],
        train_save_dir="/train14/superbrain/zlhu12/data/Download/downstream/MS/latest_3_dim/valid",
        test_save_dir="",
        n_hvg=3000,
        data_is_raw=False,
    )
    ms_valid.process_file()


    ### pert: for pert dataset the gene list for cells are the same, just keep one graph for them.
    # from gears import PertData
    # data_name = "norman"
    # split = "simulation"
    # pert_data_path = "./norman/pretrain/"
    # pert_data = PertData(pert_data_path)
    # # pert_data.load(data_name = data_name) # 91205 × 5045 89357 × 5045\
    # pert_data.load(data_path=pert_data_path)
    # pert_data.prepare_split(split=split, seed=1,train_gene_set_size=0.8,
    #                     combo_seen2_train_frac=0.8)
    # cell_graphs = pert_data.get_dataloader(batch_size=16)
    # genes = pert_data.adata.var["gene_name"].tolist()
    # genes_up = np.array([gene.upper() for gene in genes])
    # node_map = pert_data.node_map
    # print(f"genes: {genes_up}")
    # print(pert_data.adata)


    # # get corr matrix
    # corr_path = "./corr.pt"
    # if os.path.exists(corr_path):
    #     print("Already exists corr!")
    #     corr = torch.load(corr_path)
    # else:
    #     print("calculate corr...")
    #     ctrl_adata = pert_data.adata[pert_data.adata.obs['condition'] == 'ctrl']
    #     X_torch = torch.from_numpy(ctrl_adata.X.todense())
    #     corr = torch.corrcoef(X_torch.T)
    #     row, col = np.diag_indices_from(corr)
    #     corr[row,col] = 0 
    #     corr = torch.where(torch.isnan(corr), torch.full_like(corr, 0), corr)
    #     print("Corr matrix:")
    #     print(corr)
    #     print(corr.shape)
    #     torch.save(corr,corr_path)


    # edge_path = "./edge.pickle"
    # if os.path.exists(edge_path):
    #     print("Already exists edge")
    #     with open(edge_path,"rb") as f:
    #         index_edge,type_attr,corr_attr,chromo_attr=pickle.load(f)
    # else:
    #     print("Generating edge ...")
    #     pert_train=Preprocess(
    #         adata_path="",
    #         train_save_dir="",
    #         test_save_dir="",
    #         n_hvg=3550,
    #         data_is_raw=False,
    #     )
    #     index_edge,type_attr,corr_attr,chromo_attr=pert_train.generate_edge_adata(corr,genes_up,cls_append=False)
    #     with open(edge_path,"wb") as f:
    #         pickle.dump((index_edge,type_attr,corr_attr,chromo_attr),f)
    pass