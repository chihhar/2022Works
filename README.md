#############################
提案法フォルダ: /data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process
Option:
    -gene: データの種類 ex."h1","jisin","911"
    --pre_attn: PreAttention
    --phase: Phase分け学習
    --grad_log: 勾配logの保存
    
    -trainvec_num: 系列代表ベクトルの個数 
    -pooling_k: anchorベクトルの個数 
    
    -d_model: 時間エンコーディング次元
    -d_k: Key次元
    -d_v: Value次元
    -n_head: Attentionヘッド数
    -d_inner_hid: Attention後のFN層の次元
    -n_layers: Attentionの繰り返し
    -linear_num: 非線形変換の繰り返し

    -loss_scale: MAEとloglikelihood比率
    -device_num: デバイス番号

    -imp: メモ

Hawkes過程:
    学習あり:python Main.py -gene=h1 --pre_attn --phase --train
    テスト:python Main.py -gene=h1 --pre_attn --phase

カリフォルニア地震：　"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/date_jisin.90016"
    学習:python Main.py --pre_attn -gene=jisin --phase --train
    テスト:python Main.py --pre_attn -gene=jisin --phase
Emergency Call:
    #100住所すべて
    学習ありpython Main.py --pre_attn -gene=911_All --phase --train
    テスト:python Main.py --pre_attn -gene=911_All --phase
    train:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_train.pkl"
    valid:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_valid.pkl"
    test:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_All100_sliding_test.pkl"
    
    #最大の頻度
    学習あり:python Main.py --pre_attn -gene=911_1 --phase --train
    テスト:python Main.py --pre_attn -gene=911_1 --phase
    train_data:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_train.pkl"
    valid:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_valid.pkl"
    test:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_1_freq_sliding_test.pkl"
    
    #50番目の頻度
    学習ありpython Main.py --pre_attn -gene=911_50 --phase --train
    テスト:python Main.py --pre_attn -gene=911_50 --phase
    train_dat:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_train.pkl"
    valid:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_valid.pkl"
    test:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_50_freq_sliding_test.pkl"
    
    #100番目の頻度
    学習あり:python Main.py --pre_attn -gene=911_100 --phase --train
    テスト:python Main.py --pre_attn -gene=911_100 --phase
    train_data:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_train.pkl"
    valid:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_valid.pkl"
    test:"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/Call_100_freq_sliding_test.pkl"
######################################
######################################
THPフォルダ: /data1/nishizawa/Desktop/THP_compare/Transformer-Hawkes-Process
Hawkes過程
    学習:python Main.py -gene=h1  --pre_attn --train
    テスト:python Main.py -gene=h1 --pre_attn
カリフォルニア地震: 
    学習:python Main.py -gene=jisin --pre_attn --train
    テスト:python Main.py -gene=jisin --pre_attn

#############################
Main.py:訓練やテストの実行

Transformer/Model.py: 
    class Transformer:モデルの定義
Transformer/Layers.py: EncoderやDecoderLayerの定義
Transformer/SubLayers.py: SelfAttention,MultiHeadAttentionの定義
Transformer/Modules.py: Attentionの処理

Utils.py:
    def log_likelihood:対数尤度の計算
    def compute_integral_unbiased: モンテカルロ積分
    def time_loss_ae:予測値の絶対誤差の計算