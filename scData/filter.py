import pandas as pd
import anndata as ad
import pickle

def save_ms():
    # transform and filter Soriginal MS dataset.
    adata_dir="./MS"
    data_path = f"{adata_dir}/ExpDesign-E-HCAD-35.tsv"
    data = pd.read_table(data_path)
    obs = pd.DataFrame()
    var = pd.DataFrame()
    obs["cell_name"] = data['Assay']
    obs['normal_disease'] =data['Factor Value[disease]']
    obs["cell_type"] = data['Factor Value[inferred cell type - authors labels]']
    obs.replace({'cell_type':{"pyramidal neuron?":"pyramidal neuron","mixed glial cell?":"mixed glial cell"}},inplace=True)
    gene_list = f"{adata_dir}/Homo_sapiens_all_genelist.xlsx"
    gene_pd = pd.read_excel(gene_list)
    gene_ensg = pd.read_csv(f"{adata_dir}/E-HCAD-35.aggregated_filtered_counts.mtx_rows",header=None,delimiter='\t')
    gene_ensg.columns=['name','Gene stable ID']
    df_m = pd.merge(gene_ensg,gene_pd,how='outer')
    df = df_m[pd.notnull(df_m['name'])]
    var["gene_name"] = df['Gene name'].str.upper()
    data_path = f"{adata_dir}/E-HCAD-35.aggregated_filtered_counts.mtx"
    bdata = ad.read_mtx(data_path, dtype='float32')
    adata = ad.AnnData(X=bdata.X.T.todense(), obs=obs, var=var, dtype="float32")
    adata = adata[~adata.obs['cell_type'].isnull(),:]
    adata = adata[:,~adata.var['gene_name'].isnull()]
    adata_train = adata[adata.obs['normal_disease']=='normal',:]
    adata_train = adata_train[~adata_train.obs['cell_type'].isin(['B cell', 'T cell', 'oligodendrocyte B','stromal cell']),:]
    adata_test = adata[adata.obs['normal_disease']!='normal',:]
    adata_test = adata_test[~adata_test.obs['cell_type'].isin(['B cell', 'T cell', 'oligodendrocyte B','stromal cell']),:]
    with open('./ms_train.pickle','wb') as f:
        pickle.dump(adata_train,f)
    with open('./ms_test.pickle','wb') as f:
        pickle.dump(adata_test,f)