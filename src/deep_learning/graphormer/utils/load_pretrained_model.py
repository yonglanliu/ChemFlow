import torch
from src.deep_learning.utils.distributed import main_print

def load_graphormer_backbone(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Fairseq checkpoint
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # Extract only Graphormer backbone
    graph_encoder_state = {
        k.replace("encoder.graph_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.graph_encoder.")
    }

    missing, unexpected = model.load_state_dict(
        graph_encoder_state,
        strict=False,
    )

    main_print("Loaded {} parameters".format(len(graph_encoder_state)))
    main_print("Missing keys:", missing)
    main_print("Unexpected keys:", unexpected)
    return model