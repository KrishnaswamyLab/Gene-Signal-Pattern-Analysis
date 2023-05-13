for model in Signals GSPA GSPA_QR MAGIC Node2Vec_Gcell DiffusionEMD GAE_noatt_Gcell GAE_att_Gcell Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene; do
    echo ${model}
    for task in localization coexpression; do
        for run in {0..2}; do
            python train.py --model ${model} --task ${task} --save-as ${run}
	done
    done
done
