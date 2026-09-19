# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import io
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


def init_state():
    defaults = {
        "uniprot_df": pd.DataFrame(),
        "selected_uniprot_id": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def query_gene(gene, fields_string, url):
    query_string = f"gene:{gene.strip()}"

    params = {
        "query": query_string,
        "format": "tsv",
        "fields": fields_string,
    }

    try:
        response = requests.get(url, params=params, timeout=60)

        if response.status_code != 200:
            st.error(
                f"UniProt API error for {gene} "
                f"(Status {response.status_code}): {response.text}"
            )
            return pd.DataFrame()

        if len(response.text.strip().split("\n")) <= 1:
            st.info(f"No matching records found for {gene}.")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(response.text), sep="\t")
        df["query_gene"] = gene

        return df

    except Exception as e:
        st.error(f"Unexpected error for {gene}: {e}")
        return pd.DataFrame()


def show_uniprot_table(df):
    st.subheader("UniProt Results")

    event = st.dataframe(
        df,
        use_container_width=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="uniprot_results_table",
    )

    selected_rows = event.selection.rows

    if selected_rows:
        selected_df = df.iloc[selected_rows]

        if "Entry" in selected_df.columns:
            selected_uniprot_ids = selected_df["Entry"].tolist()
            st.session_state["selected_uniprot_id"] = selected_uniprot_ids

            st.success(f"Selected {len(selected_uniprot_ids)} UniProt ID(s).")
            st.write(selected_uniprot_ids)
        else:
            st.error("Column 'Entry' not found in UniProt results.")


def download_combined_table(df):
    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Combined UniProt Table",
        data=csv_data,
        file_name="combined_uniprot_results.csv",
        mime="text/csv",
        key="download_combined_uniprot",
    )


def design():
    init_state()

    st.header("Search Parameters")

    gene_input = st.text_area(
        label="Gene Names to query",
        key="gene_names",
        placeholder="Example:\nPIK3CA\nPIK3CB\nMTOR",
    )

    fields_list = [
        "accession",
        "id",
        "gene_names",
        "protein_name",
        "organism_name",
        "length",
        "sequence",
    ]

    selected_fields = st.multiselect(
        "Columns to fetch:",
        options=fields_list,
        default=fields_list,
        key="uniprot_selected_fields",
    )

    if st.button("QUERY", type="primary", key="uniprot_query"):
        url = "https://rest.uniprot.org/uniprotkb/search"

        if not gene_input:
            st.warning("Please enter at least one gene name.")
            return

        if not selected_fields:
            st.warning("Please select at least one column field.")
            return

        gene_list = [
            item.strip()
            for item in gene_input.splitlines()
            if item.strip()
        ]

        fields_string = ",".join(selected_fields)

        result_dfs = []

        with st.spinner("Fetching data from UniProt..."):
            for gene in gene_list:
                df_gene = query_gene(
                    gene=gene,
                    fields_string=fields_string,
                    url=url,
                )

                if not df_gene.empty:
                    result_dfs.append(df_gene)

        if not result_dfs:
            st.warning("No UniProt records found.")
            st.session_state["uniprot_df"] = pd.DataFrame()
            return

        combined_df = pd.concat(
            result_dfs,
            ignore_index=True,
        )

        if "Entry" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(
                subset=["Entry"],
                keep="first",
            )

        combined_df = combined_df.reset_index(drop=True)

        st.session_state["uniprot_df"] = combined_df
        st.session_state["selected_uniprot_id"] = []

    uniprot_df = st.session_state.get("uniprot_df", pd.DataFrame())

    if not uniprot_df.empty:
        st.success(f"Found {len(uniprot_df)} unique UniProt entry/entries.")
        show_uniprot_table(uniprot_df)
        download_combined_table(uniprot_df)