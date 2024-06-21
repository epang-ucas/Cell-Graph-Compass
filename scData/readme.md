# The folder contains the code files for downstream data preprocessing.
1. data_util.py: the function file of data process.
2. filter.py: transform and filter raw MS data(https://www.ebi.ac.uk/gxa/sc/experiments/E-HCAD-35. ) into train and tese splits.
3. process_downstream.py: the file for preprocessing downstream datasets. The single cell datasets will be tranformed into lmdb datasets with graph information. 