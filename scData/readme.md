This is where the open-source data for the paper "Cell-GraphCompass: Modeling Single Cells with Graph Structure Foundation Model" is located.
Cell-GraphCompass (CGCompass) is a single-cell foundational model pre-trained on fifty million human single-cell sequencing data based on graph structure. Here we open-sources the pre-trained results of CGCompass as well as the datasets for downstream tasks. To verify the effectiveness of the CGCompass methodology, we additionally pre-trained a model based solely on a transformer on the same pre-training dataset.

- Paper link: [https://www.biorxiv.org/content/10.1101/2024.06.04.597354v1](https://www.biorxiv.org/content/10.1101/2024.06.04.597354v1)
- Github link: [https://github.com/epang-ucas/Cell-Graph-Compass](https://github.com/epang-ucas/Cell-Graph-Compass)
- Google Drive link:[https://drive.google.com/drive/folders/1-0tE2jdodlUio2Wds61FKRE1E2Cd7MXU?usp=sharing]
  
Please note that only a portion of the data for this folder is stored on GitHub. The remaining data is open-sourced and available at this Google Drive link.

Directory structure:

- `./scData/process_downstream.py`: the file for preprocessing downstream datasets. The single cell datasets will be tranformed into lmdb datasets with graph information. 
- `./scData/data_util.py`: the function file of data process.
- `./scData/filter.py`: transform and filter raw MS data(https://www.ebi.ac.uk/gxa/sc/experiments/E-HCAD-35. ) into train and tese splits.
- `./scData/pretrain_weights`: Provides the pre-trained weights for CGCompass, as well as ablation experiment of the graph structure.
- `./scData/bioFeature_embs`: Here, you will find some feature data (processed) used for constructing cell graphs.
- `./scData/example_datasets`: Example databases (already processed), stored in the form of lmdb.
- `./scData/downstream_weights: The weights of CGCompass after fine-tuning for downstream tasks.

All data used for downstream tasks in this paper come from public datasets:

Datasets for cell clustering

[Perirhinal Cortex](https://drive.google.com/file/d/1rDAxDtvWx1GpJaNhlKBi71f8-psUNppE/view)

[PBMC 10K](https://docs.scvi-tools.org/en/stable/api/reference/scvi.data.pbmc_dataset.html)

[COVID-19 dataset](https://drive.google.com/file/d/1eD9LbxNJ35YUde3VtdVcjkwm-f4iyJ6x/view)

Datasets for cell type annotation

[Multiple Sclerosis (M.S.) dataset](https://drive.google.com/drive/folders/1Qd42YNabzyr2pWt9xoY4cVMTAxsNBt4v)

[Myeloid (Mye.) dataset](https://drive.google.com/drive/folders/1VbpApQufZq8efFGakW3y8QDDpY9MBoDS)

[hPancreas dataset](https://drive.google.com/drive/folders/1s9XjcSiPC-FYV3VeHrEa7SeZetrthQVV)

Datasets for single-cell perturbation prediction

[Norman dataset](https://dataverse.harvard.edu/api/access/datafile/6154020)

Before inputting to the model, this paper uniformly processed the original datasets through a consistent data processing pipeline. 
   ```shell
   bash ./exps/cellClus.sh 'MS'
   ```
Here we provide examples of processed datasets:`./scData/example_datasets`.



