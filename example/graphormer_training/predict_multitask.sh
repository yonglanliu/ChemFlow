# ===============================
# Predict using the multitask Graphormer model without calibration
# ===============================

chemflow predict graphormer \
  --input /Users/yonglanliu/Desktop/ChemFlow/dataset/ADMET/ExpansionRX/expansion_data_test_butina_split.csv \
  --structure-column SMILES \
  --task-names \
    LogD_prediction \
    LogS_prediction \
    "Log(MLM_Clint)_prediction" \
    "Log(HLM_Clint)_prediction" \
    "Log(Caco2_Papp)_prediction" \
    "Log(Caco2_ER)_prediction" \
    "Log(MPPB)_prediction" \
    "Log(MBPB)_prediction" \
    "Log(MGMB)_prediction" \
  --model-checkpoint /Users/yonglanliu/Desktop/ChemFlow/expansionrx_mtl_training/Multtask_to_OneAdaptor/checkpoints/best_model.pt \
  --batch-size 64 \
  --output /Users/yonglanliu/Desktop/ChemFlow/expansionrx_mtl_training/Multtask_to_OneAdaptor/testset_prediction_no_calibration.csv

# ===============================
# Predict using the one-task Graphormer model with calibration
# ===============================

  chemflow predict graphormer \
  --input /Users/yonglanliu/Desktop/ChemFlow/dataset/ADMET/ExpansionRX/expansion_data_test_butina_split.csv \
  --structure-column SMILES \
  --task-names \
    LogD_prediction \
    LogS_prediction \
    "Log(MLM_Clint)_prediction" \
    "Log(HLM_Clint)_prediction" \
    "Log(Caco2_Papp)_prediction" \
    "Log(Caco2_ER)_prediction" \
    "Log(MPPB)_prediction" \
    "Log(MBPB)_prediction" \
    "Log(MGMB)_prediction" \
  --model-checkpoint /Users/yonglanliu/Desktop/ChemFlow/expansionrx_mtl_training/Onetask_to_OneAdaptor/checkpoints/best_model.pt \
  --batch-size 64 \
  --output /Users/yonglanliu/Desktop/ChemFlow/expansionrx_mtl_training/Onetask_to_OneAdaptor/testset_prediction.csv \
  --calibration-file /Users/yonglanliu/Desktop/ChemFlow/expansionrx_mtl_training/Onetask_to_OneAdaptor/val_calibration.csv \
  --calibration-pairs \
    LogD:LogD_prediction \
    LogS:LogS_prediction \
    "Log(MLM_Clint):Log(MLM_Clint)_prediction" \
    "Log(HLM_Clint):Log(HLM_Clint)_prediction" \
    "Log(Caco2_Papp):Log(Caco2_Papp)_prediction" \
    "Log(Caco2_ER):Log(Caco2_ER)_prediction" \
    "Log(MPPB):Log(MPPB)_prediction" \
    "Log(MBPB):Log(MBPB)_prediction" \
    "Log(MGMB):Log(MGMB)_prediction"