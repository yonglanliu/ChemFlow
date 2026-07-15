chemflow predict graphormer \
    --input ./dataset/combined/pi3k_beta_combined.csv \
    --structure-column SMILES \
    --model-checkpoint ./graphormer_training/checkpoints/best_model.pt \
    --batch-size 64 \
    --output ./graphormer_training/predictions.csv