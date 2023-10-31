for model in Signals GSPA_QR GSPA Eigenscore GFMMD GAE_att_Ggene MAGIC DiffusionEMD Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Gcell GAE_noatt_Gcell Node2Vec_Gcell; do
    echo ${model}
    task=coexpression
    for run in 7 8 9; do
        python train.py --model ${model} --task ${task} --save-as ${run} --seed ${run} &
        python train_2_branches.py --model ${model} --task ${task} --save-as ${run} --seed ${run} &
        python train_3_branches.py --model ${model} --task ${task} --save-as ${run} --seed ${run} &
    done
done
