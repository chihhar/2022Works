#!/bin/bash

# 学習とテストを行うシェルスクリプトファイル

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

method_num=3 # FT-PP, THP, 提案法

# 提案法
for gene in '911_All' '911_1' '911_50' '911_100'# 'jisin' 'h1' 'h_fix05' 
do
    cd /data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/related_works/THP
    CUDA_VISIBLE_DEVICES=0 python THP.py --pre_attn --train -gene=${gene}
    CUDA_VISIBLE_DEVICES=0 python THP.py --pre_attn -gene=${gene}
    
    cd /data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/related_works/FT_PP
    for hazard_model in "const" "exp" "pc" "omi"
    do
        CUDA_VISIBLE_DEVICES=0 python omi.py --train=True -model=${hazard_model} -gene=${gene}
        CUDA_VISIBLE_DEVICES=0 python omi.py --train=False -model=${hazard_model} -gene=${gene}
    done

    cd /data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process
    for num_vec in 3 5 8 10 15
    do
        CUDA_VISIBLE_DEVICES=0 python Main.py --pre_attn --phase --train -trainvec_num=${num_vec} -pooling_k=${num_vec} -gene=${gene}
        CUDA_VISIBLE_DEVICES=0 python Main.py --pre_attn --phase -trainvec_num=${num_vec} -pooling_k=${num_vec} -gene=${gene}
    done
done