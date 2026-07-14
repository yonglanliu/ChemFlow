from src.deep_learning.graphormer.modules.graphormer_featurizer import GraphormerFeaturizer
from src.deep_learning.graphormer.modules.graphormer_encoder import GraphormerGraphEncoder
from src.deep_learning.graphormer.modules.graphormer_layers import GraphormerGraphEncoderLayer, GraphNodeFeature, GraphAttnBias
from src.deep_learning.graphormer.modules.dataset import GraphormerMoleculeDataset, featurize_and_cache_dataset
from src.deep_learning.graphormer.utils.load_pretrained_model import load_graphormer_backbone


from types import SimpleNamespace

# -------------------------
#         Config
# -------------------------
def get_config():
    return SimpleNamespace(
        arch="graphormer_base",
        num_target=4,

        # ----- Required Graphormer structural params -----
        max_nodes=128,
        num_atoms=512 * 9,
        num_in_degree=512,
        num_out_degree=512,
        num_edges=512 * 3,
        num_spatial=512,
        num_edge_dis=128,
        edge_type="multi_hop",
        multi_hop_max_dist=5,
        spatial_pos_max=1024,

        # ----- Encoder architecture -----
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=768,
        encoder_attention_heads=32,

        dropout=0.1,
        attention_dropout=0.1,
        act_dropout=0.0,
        encoder_normalize_before=True,
        pre_layernorm=False,
        apply_graphormer_init=False,
        activation_fn="gelu",
        tokens_per_sample=512,

        lr=1e-4,

        share_encoder_input_output_embed=False,
        no_token_positional_embeddings=False,

        pretrained_model_name="none",
        load_pretrained_model_output_layer=False,
        remove_head=True,
        use_gate_soft_sharing=True,
    )