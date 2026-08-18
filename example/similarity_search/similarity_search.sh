chemflow search similarity \
    --query_smiles 'CCCS(=O)(=O)NC1=CC=C(F)C(C(=O)C2=CNC3=NC=C(C4=CC=C(C(=O)NCC)N=C4)C=C23)=C1F' \
    --database ./dataset/ADMET/CYP/CYP_dataset.csv \
    --structure-column SMILES \
    --rep_type ecfp4 \
    --metric tanimoto \
    --num_workers 1 \
    --num_shards 2 \
    --top_k_ratio 0.1 \
    --job_name similarity_search_smiles_test

chemflow search similarity \
    --query_file './query.smi' \
    --database ./dataset/ADMET/CYP/CYP_dataset.csv \
    --structure-column SMILES \
    --rep_type ecfp4 \
    --metric tanimoto \
    --num_workers 1 \
    --num_shards 2 \
    --top_k_ratio 0.1 \
    --job_name similarity_search_file_test