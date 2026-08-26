chemflow predict graphormer \
    --input "/Users/yonglanliu/Desktop/ChemFlow/dataset/ADMET/ExpansionRX/log_transformed_test.csv" \
    --structure-column SMILES \
    --task-names LogD_pred \
    --model-checkpoint ./graphormer_training_regression/checkpoints/best_model.pt \
    --batch-size 64 \
    --output ./graphormer_training_regression/predictions.csv