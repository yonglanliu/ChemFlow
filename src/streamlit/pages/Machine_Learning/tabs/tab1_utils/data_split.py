# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import streamlit as st


def design(workdir, task_type):

    if task_type == "classification":
        split_options = [
            "random",
            "scaffold",
            "cluster",
            "butina",
        ]

    elif task_type == "regression":
        split_options = [
            "random",
            "stratified",
            "scaffold",
            "cluster",
            "butina",
        ]

    else:
        raise ValueError(
            f"Unknown task_type: {task_type}"
        )
    c1, c2, c3, c4 = st.columns(4,vertical_alignment="bottom")
    with c1:
        split_method = st.selectbox(
            "Split Method",
            split_options,
            index=0,
            key="ml_split_method",
            help="""
            random: Random train/test split

            stratified: Preserve class distribution (classification only)

            time: Split by chronological order

            scaffold: Murcko scaffold split

            cluster: Agglomerative clustering split using Morgan fingerprints

            butina: Butina clustering split using Morgan fingerprints and Tanimoto similarity
            """,
        )

    with c2:
        test_size = st.slider(
            "Test Size",
            min_value=0.10,
            max_value=0.50,
            value=0.20,
            step=0.05,
            key="ml_test_size",
        )
    with c3:
        validation_size = st.slider(
            "Validation Size",
            min_value=0.00,
            max_value=0.30,
            value=0.0,
            step=0.05,
            key="ml_validation_size",
        )
    with c4:
        data_split_seed = st.number_input(
            "Data Split Seed",
            min_value=0,
            value=42,
            step=1,
            key="ml_data_split_seed",
            help="Random seed used for train/test splitting.",
        )


    split_config = {}

    if split_method == "cluster":
        split_config["n_clusters"] = st.number_input(
            "Number of Clusters",
            min_value=2,
            max_value=200,
            value=20,
            step=1,
            help="Number of Agglomerative clusters.",
        )

    elif split_method == "butina":
        split_config["butina_cutoff"] = st.slider(
            "Butina Distance Cutoff",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help=(
                "Distance cutoff used by Butina clustering. "
                "Distance = 1 - Tanimoto similarity. "
                "0.4 corresponds to similarity >= 0.6."
            ),
        )
    if split_method in ["cluster", "butina"]:
        c21, c22 = st.columns(2, vertical_alignment="bottom")
        with c21:
            
                split_config["fp_radius"] = st.number_input(
                    "Morgan Radius",
                    min_value=1,
                    max_value=4,
                    value=2,
                    step=1,
                )
        with c22:
            split_config["fp_n_bits"] = st.selectbox(
                "Fingerprint nBits",
                [1024, 2048, 4096],
                index=1,
            )
    c11, c12, c13 = st.columns((0.4, 1, 1), vertical_alignment="bottom")
    with c11:
        save_split_dataset = st.selectbox(
            "Save Split Dataset",
            ["Yes", "No"],
            index=0,
            key="ml_save_split_dataset",
            help="Whether to save the split dataset.",
        )
    with c12:
        if save_split_dataset == "Yes":
            save_split_dataset_path = st.text_input(
                "Save Directory",
                value=f"{workdir}/split_dataset",
                key="ml_save_split_dataset_path",
                help="Path to save the split dataset.",
            )
        else:
            save_split_dataset_path = None
    with c13:
        if save_split_dataset == "Yes":
            prefix = st.text_input("Prefix for Split Dataset Files", 
                          value=f"{split_method}_split_seed{data_split_seed}", 
                          key="ml_save_split_dataset_prefix", 
                          help="Prefix for the split dataset files.")
        else:
            prefix = None

    split_config["task_type"] = task_type
    split_config["split_method"] = split_method
    split_config["test_size"] = test_size
    split_config["validation_size"] = validation_size if validation_size > 0 else None
    split_config["random_seed"] = data_split_seed
    split_config["save_split_data"] = save_split_dataset
    split_config["save_dir"] = save_split_dataset_path
    split_config["prefix_name"] = prefix

    return split_config