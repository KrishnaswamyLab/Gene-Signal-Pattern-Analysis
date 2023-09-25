#for model in GSPA_QR GSPA Eigenscore GAE_att_Gcell GAE_noatt_Gcell Signals GFMMD GAE_att_Ggene MAGIC Node2Vec_Gcell DiffusionEMD Node2Vec_Ggene GAE_noatt_Ggene; do
for model in DiffusionEMD Node2Vec_Ggene GAE_noatt_Ggene; do
    echo ${model}
    task=coexpression
    for run in 0 1 2; do
        python train_2_branches.py --model ${model} --task ${task} --save-as ${run}
        python train_3_branches.py --model ${model} --task ${task} --save-as ${run}
    done
done
