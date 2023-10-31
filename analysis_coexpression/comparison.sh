for model in Signals DiffusionEMD GSPA GSPA_QR MAGIC Node2Vec_Gcell GAE_noatt_Gcell GAE_att_Gcell Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene; do
    echo ${model}
    for dataset in linear 2_branches 3_branches; do
        python evaluate_demap.py ${model} ${dataset}
        python evaluate_stratified_coexpression.py ${model} ${dataset}
    done
done