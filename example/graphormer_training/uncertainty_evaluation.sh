CHEMFLOW_BIN="${CHEMFLOW_BIN:-/Users/yonglanliu/Desktop/ChemFlow/.venv/bin/chemflow}"

"$CHEMFLOW_BIN" uncertainty bootstrap \
    --input /Users/yonglanliu/Desktop/ChemFlow/expansionrx_mtl_training/Onetask_to_OneAdaptor/testset_prediction_no_calibration.csv \
    --task "LogD:LogD:LogD_prediction" \
    --task "LogS:LogS:LogS_prediction" \
    --task "Log(MLM_Clint):Log(MLM_Clint):Log(MLM_Clint)_prediction" \
    --task "Log(HLM_Clint):Log(HLM_Clint):Log(HLM_Clint)_prediction" \
    --task "Log(Caco2_Papp):Log(Caco2_Papp):Log(Caco2_Papp)_prediction" \
    --task "Log(Caco2_ER):Log(Caco2_ER):Log(Caco2_ER)_prediction" \
    --task "Log(MPPB):Log(MPPB):Log(MPPB)_prediction" \
    --task "Log(MBPB):Log(MBPB):Log(MBPB)_prediction" \
    --task "Log(MGMB):Log(MGMB):Log(MGMB)_prediction" \
    --metrics r2 mae rmse pearson spearman kendall \
    --n-bootstrap 2000 \
    --confidence-level 0.95 \
    --seed 42 \
    --plot-distributions \
    --output-dir /Users/yonglanliu/Desktop/ChemFlow/expansionrx_mtl_training/Onetask_to_OneAdaptor/bootstrap_results