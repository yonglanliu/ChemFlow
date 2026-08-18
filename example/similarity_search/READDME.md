```bash
--query_smiles
    Query molecule provided as a SMILES string.

--database
    Molecular database used for similarity searching.

--structure-column
    Column in the database containing molecular structures/SMILES.

--rep_type
    Molecular representation used for similarity calculation.

    Supported representations:
        ecfp4
        ecfp6
        fcfp4
        fcfp6
        maccs
        2d_descriptor

    ECFP/FCFP fingerprints use 2048 bits.
    MACCS uses 167 bits.

--metric
    Similarity or distance metric.

    Supported metrics:
        tanimoto
        cosine
        dice
        euclidean
        manhattan
        mcconnaughey

--num_workers
    Number of worker processes used to run jobs in parallel.
    Increase this value for large datasets when multiple CPU resources
    are available.

--num_shards
    Number of subsets used to partition the original database.
    The number of shards can be greater than the number of workers.

--top_k_ratio
    Fraction of the highest-ranked results to retain.
    For example, 0.1 keeps the top 10% of results.

--job_name
    Name of the similarity-search job. This can also be used as a prefix
    for intermediate files, worker outputs, and log files.
```