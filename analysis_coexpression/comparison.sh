#for model in GFMMD Eigenscore Signals DiffusionEMD GSPA GSPA_QR MAGIC Node2Vec_Gcell GAE_noatt_Gcell GAE_att_Gcell Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene; do
for model in GAE_att_Gcell; do
    echo ${model}
    python evaluate_stratified_coexpression.py ${model}
done
