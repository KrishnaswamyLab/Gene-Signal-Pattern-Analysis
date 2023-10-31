for model in GFMMD Eigenscore GAE_att_Ggene Signals GSPA GSPA_QR MAGIC Node2Vec_Gcell DiffusionEMD GAE_noatt_Gcell GAE_att_Gcell Node2Vec_Ggene GAE_noatt_Ggene; do
    echo ${model}
    for dataset in linear 2_branches 3_branches; do
        python evaluate_localization_embedding.py ${model} ${dataset} &
        python evaluate_localization_uniform.py ${model} ${dataset} &
    done
done