from matplotlib import pyplot as plt
import torch
import numpy as np
import pdb
import transformer.Models as trm
from tqdm import tqdm
import Main
import math
import Utils
def get_generate(data_type):
    if data_type == 'sp':
        return Main.generate_stationary_poisson()
    elif data_type == 'nsp':
        return Main.generate_nonstationary_poisson()
    elif data_type == 'sr':
        return Main.generate_stationary_renewal()
    elif data_type == 'nsr':
        return Main.generate_nonstationary_renewal()
    elif data_type == 'sc':
        return Main.generate_self_correcting()
    elif data_type == 'h1':
        return Main.generate_hawkes1()
    elif data_type == 'h2':
        return Main.generate_hawkes2()
def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

###
#縦軸経過時間 横軸EventIDのplotなど
###
def Compare_event_GT_pred(model, test_data, opt):
    model.eval()
    GT_his=[]
    pred_his=[]

    with torch.no_grad():
        

        for batch in tqdm(test_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            event_time = batch.to(opt.device, non_blocking=True)
            test_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            GT_his = np.append(GT_his,np.array(test_target.squeeze(-1).cpu()))
            
            _, pred, enc_out = model(test_input,test_target)
            pred = pred.reshape(test_target.shape)
            pred_his=np.append(pred_his,pred.cpu())
            """
            if model.method=="both_scalar":
                non_pad_mask = get_non_pad_mask(torch.cat((test_input,torch.ones((test_input.shape[0],model.rep_vector.shape[1]),device=test_input.device)),dim=1))

                rep_batch = model.rep_vector.repeat([enc_out.shape[0],1])
                
                label_input=torch.cat([test_input,rep_batch] ,dim=1)

                tem_enc = model.encoder.temporal_enc(test_input)
                tem_rep = model.encoder.temporal_enc(rep_batch)
                tem_enc = torch.cat((tem_enc,tem_rep), dim=1)#(16,seqence-1,M)->(16,seq-1+trainvec,M)
                slf_attn_mask_subseq = get_subsequent_mask(test_input)
                slf_attn_mask_subseq=torch.cat((torch.cat((slf_attn_mask_subseq,torch.zeros((test_input.shape[0],model.rep_vector.shape[1],test_input.shape[1]),device=test_input.device)),dim=1),torch.zeros((test_input.shape[0],model.rep_vector.shape[1]+test_input.shape[1],model.rep_vector.shape[1]),device=test_input.device)),dim=2)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=torch.cat((test_input,torch.ones((test_input.shape[0],model.rep_vector.shape[1]),device=test_input.device)),dim=1), seq_q=torch.cat((test_input,torch.ones((test_input.shape[0],model.rep_vector.shape[1]),device=test_input.device)),dim=1))
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

                enc_output = torch.zeros(tem_enc.shape,device=test_input.device)
                plt.figure(figsize=(33,8))
                for enc_layer in model.encoder.layer_stack:
                    enc_output += tem_enc
                    enc_output, attn = enc_layer(
                        enc_input=enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask)
                    plt.clf()
                    batch_look_num=0
                    for i in range(attn.shape[1]):
                        ax = plt.subplot(4,2,i+1)
                        plt.pcolor(attn[batch_look_num,i,-model.rep_vector.shape[1]:].cpu(),cmap="gray")
                        group=[str(round(label_input[batch_look_num,j].item(),3)) for j in range(0,attn.shape[3])] 
                        for s in range(1,model.rep_vector.shape[1]+1):
                            group[-s]="S"+str(model.rep_vector.shape[1]+1-s)+":"+group[-s]
                        plt.colorbar()
                        #plt.yticks(range(attn.shape[3]),group)
                    pdb.set_trace()
                    
                    plt.savefig("test.png")
                    plt.savefig("plot/kmean_anc_gt_pred/attn/"+opt.method+opt.imp+"_"+str(pred[0].item())+".png")

                    mat = attn[batch_look_num,0,-model.rep_vector.shape[1]:,:].sum()
            """
            """        
            #anchor_batch = model.ancarvector.repeat([enc_out.shape[0],1,1])
            #x_tem_enc = model.decoder.temporal_enc(anchor_batch)
            #x_tem_enc = anchor_batch
            
            anchor_batch = model.anchor_vector.repeat([enc_out.shape[0],1])
            x_tem_enc = model.decoder.temporal_enc(anchor_batch)
            output = torch.zeros(x_tem_enc.shape,device=model.decoder.device)
            for dec_layer in model.decoder.layer_stack:
                output += x_tem_enc #residual
                output, attn = dec_layer(
                    output,
                    enc_out,
                    enc_out,
                    non_pad_mask=None,
                    slf_attn_mask=None)
                plt.clf()
                
                for i in range(attn.shape[1]):
                    plt.subplot(4,2,i+1)
                    plt.pcolor(attn[0,i].cpu(),cmap='gray')
                plt.colorbar()
                plt.savefig("plot/kmean_anc_gt_pred/attn/"+opt.method+opt.imp+"_"+str(pred[0].item())+".png", bbox_inches='tight', pad_inches=0)

                #attn[0,:,0].max(); attn[0,:,1].max();attn[0,:,2].max();attn[0,:,3].max();attn[0,:,4].max()
            
            """
    print("Compare event and GT")
    if opt.gene=="h1":
        print("Hawkes")
        #GT_his
        FTPPh1=pred_his
        THPh1=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/npy_Matome/THPh164pre_l4h1_pred.npy")
        omih1=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/npy_Matome/omih164pred.npy")
        plt.figure(figsize=(20,4))
        plt.xlabel("event ID", fontsize=18)
        plt.ylabel("elapsed time", fontsize=18)
        plt.plot(range(150),GT_his[350:500],c="m",label="ground-truth")
        plt.plot(range(150),omih1[350:500],c="r",label="FT-PP",linestyle="dotted")
        plt.plot(range(150),THPh1[350:500],c="g",label="THP",linestyle="dashdot")
        plt.plot(range(150),FTPPh1[350:500],c="b",label="Transformer-FT-PP",linestyle="dashed")
        plt.legend(fontsize=18, loc='upper right')
        print("not plot event and predictions of all method")
        plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+str(opt.n_layers)+"all_methodh1.svg", bbox_inches='tight', pad_inches=0)
        plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+str(opt.n_layers)+"all_methodh1.pdf", bbox_inches='tight', pad_inches=0)
        plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+str(opt.n_layers)+"all_methodh1.png", bbox_inches='tight', pad_inches=0)
        
        
        plt.clf()
        plt.figure(figsize=(10,10))
        omih1=omih1.reshape(FTPPh1.shape)
        Omi_AE=abs(GT_his-omih1)
        THP_AE=abs(GT_his-THPh1)
        TFTPP_AE=abs(GT_his-FTPPh1)
        points = (GT_his,Omi_AE, THP_AE,TFTPP_AE)
        fig, ax = plt.subplots()
        bp = ax.boxplot(points, showmeans=True, sym="")
        ax.set_xticklabels(['ground-truth','FT-PP', 'THP', 'proposed method'])
        plt.xlabel('methods')
        plt.ylabel('point')
        plt.grid()
        plt.savefig("h1testGT.png", bbox_inches='tight', pad_inches=0)
    elif opt.gene=="jisin":
        print("synthetic")
        if opt.method=="abstdy_nonrep3":
            np.save("npy_Matome/abstdy_nonrep3.npy",pred_his)
            #np.save("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/npy_Matome/predplot_64pre3_3_ph_l4",pred_his)
        rep3_his=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/npy_Matome/abstdy_nonrep3.npy")
        FT_PP_pred=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/npy_Matome/predplot_64pre3_3_ph_l4.npy")
        plt.clf()
        #plt.figure(figsize=(20,4))
        plt.xlabel("event ID", fontsize=18)
        plt.ylabel("elapsed time", fontsize=18)
        plt.plot(range(150),GT_his[350:500],c="m",label="ground-truth")
        plt.plot(range(150),rep3_his[350:500],c="r",label="last3",linestyle="dashdot")
        plt.plot(range(150),FT_PP_pred[350:500],c="b",label="rep_vector3",linestyle="dashed")
        #ld_THPl2=np.load("")
        #old_THPl4=np.load("")
        plt.legend(fontsize=18, loc='upper right')
        #plt.savefig("plot/abstdy/l3andrepvec"+opt.imp+"test.png", bbox_inches='tight', pad_inches=0)
        #plt.savefig("plot/abstdy/l3andrepvec"+opt.imp+"test.pdf", bbox_inches='tight', pad_inches=0)
        #plt.savefig("plot/abstdy/l3andrepvec"+opt.imp+"test.svg", bbox_inches='tight', pad_inches=0)
        TPG=abs(GT_his-FT_PP_pred)
            
        L3_Error = abs(GT_his-rep3_his)
        TFTPP_Error = abs(GT_his-FT_PP_pred)
        len=TFTPP_Error.shape[0]
        hako=np.array([0.25,0.5,0.75])
        TFTPP_Error.sort()
        L3_Error.sort()
            
        print("AE点25-50-75直近３つの変換\n"+str(L3_Error[(len*hako).astype(int)]))
        print("AE点25-50-75提案法:\n"+str(TFTPP_Error[(len*hako).astype(int)]))


        leng=GT_his.shape[0]
        #hako=np.array([0.25,0.5,0.75,0.8,0.9,0.975,0.99])
        GT_his[(leng*hako).astype(int)]
        hako=np.array([0.25,0.5,0.75,0.8,0.9,0.975,0.99])
        #plot GT and pred
        plt.clf()#kakutei64pre_l4///_encmlnnasi
        THP_pred=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/npy_Matome/THPkakutei64pre_l4_encmlnnasijisin_pred.npy")#kakutei64pre_l4
        FT_PP_pred=pred_his#np.load("Transformer_FT-PP64jisin_tau_pred.npy")
        omi_pred=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/FT-PP_tau_pred.npy")
        #plt.figure(figsize=(20,4))
        print("evemt ID:横軸")
        plt.xlabel("event ID", fontsize=18)
        print("elapsed time:縦軸")
        plt.ylabel("elapsed time", fontsize=18)
        plt.plot(range(150),GT_his[350:500],c="m",label="ground-truth")
        plt.plot(range(150),omi_pred[350:500],c="r",label="FT-PP",linestyle="dotted")
        plt.plot(range(150),THP_pred[350:500],c="g",label="THP",linestyle="dashdot")
        plt.plot(range(150),FT_PP_pred[350:500],c="b",label="Transformer-FT-PP",linestyle="dashed")
        plt.legend(fontsize=18, loc='upper right')
        print("ploted event and predictions of all method,THP,FT-PP,TransformerFT-PP")
        #plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+str(opt.n_layers)+"all_methodjisinnonLN.svg", bbox_inches='tight', pad_inches=0)
        #plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+str(opt.n_layers)+"all_methodjisinnonLN.pdf", bbox_inches='tight', pad_inches=0)
        #plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+str(opt.n_layers)+"all_methodjisinnonLN.png", bbox_inches='tight', pad_inches=0)
        #pdb.set_trace()
        """
        plt.clf()
        plt.figure(figsize=(10,4)) 
        plt.xlabel("event ID",fontsize=18)
        plt.ylabel("elapsed time",fontsize=18)
        plt.plot(range(100),GT_his[200:300],label="ground-truth")
        plt.plot(range(100),pred_his[200:300],c="r",label="k-means anchor",linestyle="dashed")
        
        plt.legend(fontsize=18, loc='upper right')
        
        #plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+opt.method+opt.imp+".png", bbox_inches='tight', pad_inches=0)
        #plt.savefig("plot/kmean_anc_gt_pred/IDtime/"+opt.method+opt.imp+".svg", bbox_inches='tight', pad_inches=0)
        #np.save("THP_pred_his_64_jisin.npy",pred_his)
        #np.save("THP_pred_his_64preh1l4.npy",pred_his)
        #np.save("Transformer_FT-PPh164pre33phl4_tau_pred.npy",pred_his)
        #np.save("Transformer_FT-PP64jisin_tau_pred.npy",pred_his)
        #np.save("THP_pred_his_64_jisin.npy",pred_his)
        """
        #THP_pred=np.load("THP_pred_his_64_jisin.npy")
        #FT_PP_pred=np.load("Transformer_FT-PP64jisin_tau_pred.npy")
        #omi_pred=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/data/FT-PP_tau_pred.npy")
        ##/plot GT and pred
        omi_pred=omi_pred.reshape(FT_PP_pred.shape)
        Omi_AE=abs(GT_his-omi_pred)
        THP_AE=abs(GT_his-THP_pred)
        TFTPP_AE=abs(GT_his-FT_PP_pred)
        plt.clf()
        print("GT-pred scatter")
        #plt.figure(figsize=(10,10))
        plt.scatter(GT_his,omi_pred,alpha=0.5,color="r",label="FT-PP")
        plt.scatter(GT_his,THP_pred,alpha=0.5,color="g",label="THP")
        plt.scatter(GT_his,FT_PP_pred,alpha=0.5,color="b",label="Transformer-FT-PP")
        plt.xlabel("GT")
        plt.ylabel("prediction")
        plt.legend()
        plt.savefig("dalljisin_sct.png", bbox_inches='tight', pad_inches=0)
        plt.savefig("dalljisin_sct.svg", bbox_inches='tight', pad_inches=0)
        
        plt.clf()
        len=TFTPP_AE.shape[0]
        hako=np.array([0.25,0.5,0.75])
        Omi_AE.sort()
        THP_AE.sort()
        TFTPP_AE.sort()
        
        print("FT-PP:"+str(Omi_AE[(len*hako).astype(int)]))
        print("THP:"+str(THP_AE[(len*hako).astype(int)]))
        print("提案:"+str(TFTPP_AE[(len*hako).astype(int)]))
        
        #箱ひげ図
        print("箱ひげplot")
        plt.clf()
        points = (GT_his,Omi_AE, THP_AE,TFTPP_AE)
        fig, ax = plt.subplots()
        bp = ax.boxplot(points, showmeans=True, sym="")
        ax.set_xticklabels(['ground-truth','FT-PP', 'THP', 'proposed method'])
        plt.xlabel('methods')
        plt.ylabel('point')
        plt.grid()
        plt.savefig("ctestGT.svg", bbox_inches='tight', pad_inches=0)
"""
plt.clf()
plt.figure(figsize=(10,4))
plt.xlabel("event ID", fontsize=18)
plt.ylabel("elapsed time", fontsize=18)
plt.plot(range(100),GT_his[200:300],label="ground-truth")
plt.plot(range(100),omi_pred[200:300],c="r",label="FT-PP",linestyle="dashed")
plt.plot(range(100),THP_pred[200:300],c="g",label="THP",linestyle="dashed")
plt.plot(range(100),FT_PP_pred[200:300],c="b",label="Transformer-FT-PP",linestyle="dashed")
plt.legend(fontsize=18, loc='upper right')
plt.savefig("plot/kmean_anc_gt_pred/IDtime/all_methodjisin.pdf", bbox_inches='tight', pad_inches=0)
plt.savefig("plot/kmean_anc_gt_pred/IDtime/all_methodjisin.png", bbox_inches='tight', pad_inches=0)
"""
def save_npy_synthetic(model, testloader, opt):
    """
    synthetic_dataのGT_intensityとhat_hazardを比較するもの
    
        hat_hazardをnpy形式にて保存する関数
    Args:
        
        ...

    Vars
        
    """
    
    model.eval()
    #select data:
    test_data=testloader.__iter__()
    test_datax = test_data.next()[1:10]
    
    #prepare data:
    event_time = test_datax.to(opt.device)
    input = event_time[:,:-1]
    target = test_datax[:,-1:]
    [T,score]=get_generate(opt.gene)
    test_datat=T[80000:]
    dT_test = np.ediff1d(test_datat)
    rt_test = torch.tensor(rolling_matrix(dT_test,opt.time_step)).to(torch.double)

    t_min=0
    t_max=target.sum()+math.pow(10,-9)
    loop_start=0
    loop_num=5000
    loop_delta = (t_max-t_min)/loop_num
    print_progress_num = 1000
    
    cumsum_tau = torch.cumsum(target,dim=0).to(opt.device)
    log_likelihood_history = []
    non_log_likelihood_history = []
    target_history = []
    calc_log_l_history = []
    with torch.no_grad():
        for t in range(loop_start,loop_num):
            if t % print_progress_num == 0:
                print(t)
            now_row_number = (target.size(0) - ( cumsum_tau > t*loop_delta+math.pow(10,-9)).sum().item())
            if now_row_number >= target.size(0):
                break
            
            now_input = input[now_row_number:now_row_number+1]
            now_target = target[now_row_number:now_row_number+1] 
            
            minus_target_value = cumsum_tau[now_row_number-1] if now_row_number >0 else 0
            variation_target = torch.tensor((t*loop_delta+math.pow(10,-9)),device=input.device)- minus_target_value
            
            variation_target = variation_target.reshape(now_target.shape)
            output, prediction, enc_out = model(now_input,variation_target)
            event_ll, non_event_ll = Utils.log_likelihood(model, output, now_input, variation_target,enc_out)            
            
            all_t = T[80000+opt.time_step]+t*loop_delta+math.pow(10,-9)
            if opt.gene =="sp":
                log_l_t = np.log(1)
            elif opt.gene =="nsp":
                log_l_t = np.log(0.99*np.sin((2*np.pi*all_t.cpu().numpy())/20000)+1)
            elif opt.gene=="h1":
                log_l_t = np.log(0.2 + (0.8*np.exp(-(all_t.cpu().numpy() - T[T<all_t.cpu().numpy()]))).sum())
            elif opt.gene=="h2":
                log_l_t = np.log(0.2 + (0.4*np.exp(-(all_t.cpu().numpy()-T[T<all_t.cpu().numpy()]))).sum() + (0.4*20*np.exp(-20*(all_t.cpu().numpy()-T[T<all_t.cpu().numpy()]))).sum())
            elif opt.gene=="sc":
                past_event_num = ((T<all_t.cpu().numpy()).sum())
                
                log_l_t = np.log(np.exp(all_t.cpu().numpy() - past_event_num))
            elif opt.gene=="sr":
                log_l_t = 0
            elif opt.gene=="nsr":
                log_l_t=0  

            calc_log_l_history = np.append(calc_log_l_history,log_l_t)
            log_likelihood_history = np.append(log_likelihood_history,event_ll.cpu().detach().numpy())
            non_log_likelihood_history =np.append(non_log_likelihood_history,non_event_ll.cpu().detach().numpy())
            target_history+=[t*loop_delta+math.pow(10,-9)]
    #np.save("npy_Matome/"+opt.method+opt.imp+opt.gene+"_calc_intensity.npy",log_likelihood_history)
    #np.save("npy_Matome/GT_intensity.npy",calc_log_l_history)
    #np.save("npy_Matome/target_history.npy",target_history)

    plt.clf()
    plt.figure(figsize=(10,10))
    plt.plot(target_history,calc_log_l_history,label=r"ground-truth",color="r")
    plt.scatter(cumsum_tau.cpu(),torch.zeros(cumsum_tau.shape)-2,marker='x',color="k",label="event-time")
    #THPSLOG=np.load("THP.npy")
    THP_ll=np.load("/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/npy_Matome/THPh164pre_l4h1_calc_intensity.npy")
    plt.plot(target_history,THP_ll,label=r"THP",color="b",linestyle="dashdot")
    #plt.plot(target_history,THPSLOG,label=r"THP",linestyle="dashed")
    plt.plot(target_history,log_likelihood_history,label=r"proposed method",color="g",linestyle="dashed")
    plt.ylim(-2.5,1.5)
    plt.xlabel(r"time", fontsize=18)
    plt.ylabel(r"log-intensity", fontsize=18)
    #plt.title("toy data", fontsize=18)
    plt.legend(fontsize=18)
    #pdb.set_trace()
    #for i in range(input.shape[0]-1):
    #    plt.scatter(input[i+1].cpu()+cumsum_tau.cpu()[i],np.zeros(opt.time_step-1),color="y")
    #plt.plot(cumsum_tau.cpu(),to_plot_log_l.cpu(),label="true_log_l")
    #plt.plot(cumsum_tau.cpu(),to_plot_Int_l.cpu(),label="true_Int_l")
    #plt.xlabel("Time t")
    #plt.ylabel(r"Conditional_intensity log $\lambda(t|H_t)$")
    
    #plt.rc("svg", fonttype="none")
    print("atest")
    plt.savefig("atest.svg",bbox_inches='tight', pad_inches=0)
    pdb.set_trace()

    print("end")
def near_tau_and_vector(model,opt):
    print("系列代表ベクトルに近いtauの探索(廃止)")
    #pdb.set_trace()
    #Division_Num=int(opt.train_max*4)
    Division_Num=50000
    """
    Division_time=[opt.train_max / Division_Num * i for i in range(Division_Num)]
    tensor_time=torch.tensor(Division_time).to(opt.device).unsqueeze(0)
    encoded_time=model.encoder.temporal_enc(tensor_time)
    
    rep_near=np.zeros([opt.trainvec_num,Division_Num])
    anchor_near=np.zeros([opt.pooling_k,Division_Num])
    
    for rep_n in range(opt.trainvec_num):
        #rep_near[rep_n]=np.argsort((torch.cosine_similarity(model.train_parameter[:,rep_n,:],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
        rep_near[rep_n]=np.argsort((torch.cosine_similarity(model.rep_vector[:,rep_n,:],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
    for anc_n in range(opt.pooling_k):
        #anchor_near[anc_n]=np.argsort((torch.cosine_similarity(model.ancarvector[:,anc_n,:],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
        anchor_near[anc_n]=np.argsort((torch.cosine_similarity(model.anchor_vector[:,anc_n],encoded_time[0,:,:])).cpu().detach().numpy())[::-1]
    rep_near=rep_near.astype(int)
    anchor_near=anchor_near.astype(int)
    pdb.set_trace()
    [((torch.cosine_similarity(model.anchor_vector[:,0,:],encoded_time[0,:,:])).cpu().detach().numpy()).max(),((torch.cosine_similarity(model.anchor_vector[:,1,:],encoded_time[0,:,:])).cpu().detach().numpy()).max(),((torch.cosine_similarity(model.ancarvector[:,2,:],encoded_time[0,:,:])).cpu().detach().numpy()).max()]
    print((np.array(Division_time)[rep_near])[:,:20])
    print((np.array(Division_time)[anchor_near])[:,:20])
    """
    #torch.cosine_similarity(model.ancarvector[:,:,:],model.ancarvector[:,::])  
def synthetic_plot(model, plot_data, opt):
    print("synthetic plot")
    print("要 改善")
    with torch.no_grad():
        motonotime=[opt.train_max / 1000 * i for i in range(1000)]
        tensortime=torch.tensor(motonotime).to(opt.device).unsqueeze(0)
        temptime=model.encoder.temporal_enc(tensortime)
        event_num=0
        ae=0
        all_num=0
        """
        for tn in range(opt.trainvec_num):
            cost0=np.argsort((torch.cosine_similarity(model.[:,tn,:],temptime[0,:,:])).cpu().detach().numpy())[::-1]
            print((np.array(motonotime)[cost0])[:10])
                   
        for an in range(opt.pooling_k):
            anc0=np.argsort((torch.cosine_similarity(model.ancarvector[:,an,:],temptime[0,:,:])).cpu().detach().numpy())[::-1]  
            print((np.array(motonotime)[anc0])[:10])
        """
        target_history=[]
        prediction_history=[]
        LLpred_history=[]
        for batch in tqdm(plot_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            train_input = event_time[:,:-1]
            train_target = event_time[:,-1:]
            
            target_history=np.append(target_history,train_target.cpu())
            """ forward """
            
            model_output, prediction,enc_out = model(train_input,train_target)
            
            #LLpred=Utils.time_mean_prediction(model,model_output,train_input,train_target,enc_out,opt)
            #LLpred_history=np.append(LLpred_history,LLpred.cpu())
            prediction = prediction.reshape(train_target.shape)
            prediction_history=np.append(prediction_history,prediction.cpu())
            ae+=np.abs(prediction.cpu() - train_target.cpu()).sum()
            
            event_num+=event_time.shape[0]
            plt.scatter(train_target.cpu(),prediction.cpu(),c='blue')                
        
        gosa=abs(target_history-prediction_history)
        len=gosa.shape[0]
        gosa.sort()
        hako=np.array([0.25,0.5,0.75])
        print(gosa[(len*hako).astype(int)])
        
        print(ae/event_num)
        #print((abs(target_history-LLpred_history).sum())/event_num)
        plt.xlabel('event index')
        plt.ylabel('prediction')
        plt.title('toy data')
        plt.savefig("plot/syn_tru/THP_event_index"+opt.imp+'.png', bbox_inches='tight', pad_inches=0)
        
        plt.clf()
        plt.xlabel(r'elapsed time$\tau$')
        plt.ylabel('count')
        #plt.title('tau count histgram')
        hist_min = np.append(prediction_history,target_history).min()
        hist_max = np.append(prediction_history,target_history).max()
        bins = np.linspace(hist_min, hist_max, 100)
        plt.ylim(0,2000)
        
        plt.hist([target_history,prediction_history],range=(0,3), bins=100, alpha = 0.5, label=['True',"Prediction"])
        #plt.hist(prediction_history, bins,alpha = 0.5, label='b')
        plt.legend()
        plt.rc("svg", fonttype="none")
        plt.savefig("plot/syn_hist/hist"+opt.gene+'_'+opt.imp+'_'+str(opt.time_step)+"_"+str(opt.epoch)+'.svg', bbox_inches='tight', pad_inches=0)
        plt.savefig("plot/syn_hist/hist"+opt.gene+'_'+opt.imp+'_'+str(opt.time_step)+"_"+str(opt.epoch)+'.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        
        plt.figure(figsize=(10,4))
        plt.xlabel("event iD",fontsize=18)
        plt.ylabel("elapsed time",fontsize=18)
        
        plt.plot(range(100),target_history[200:300],label="ground-truth")
        plt.plot(range(100),prediction_history[200:300],c="r",label="pred",linestyle="dashed")
        plt.legend(fontsize=18, loc='upper right')
        #plt.rc("svg", fonttype="none")
        plt.savefig("plot/syn_hist/ID_time"+opt.imp+".svg", bbox_inches='tight', pad_inches=0)
        plt.savefig("plot/syn_hist/seismic_id_timetau_hat_flatafnorm5_5+ID_time"+opt.imp+".png", bbox_inches='tight', pad_inches=0)
        print(np.sqrt((((target_history-target_history.mean())**2).sum())/target_history.shape[0]))
        gosa=abs(prediction_history-target_history)
        
        len=gosa.shape[0]
        gosa.sort()
        hako=np.array([0.25,0.5,0.75])
        print(gosa[(len*hako).astype(int)])
        print(np.sqrt((((gosa-gosa.mean())**2).sum())/gosa.shape[0]))
        
        print(np.sqrt((((prediction_history-prediction_history.mean())**2).sum())/prediction_history.shape[0]))
        sxx=(((target_history-target_history.mean())**2).sum())/target_history.shape[0]  
        syy=(((prediction_history-prediction_history.mean())**2).sum())/prediction_history.shape[0]
        sxy=(((target_history-target_history.mean())*(prediction_history-prediction_history.mean())).sum())/prediction_history.shape[0]
        corr=sxy/(np.sqrt(sxx)*np.sqrt(syy))
        
        print(corr)     
def plot_learning_curve(train_loss_his,valid_loss_his,opt):
    plt.clf()
    plt.plot(range(len(train_loss_his)),train_loss_his,label="train_curve")
    plt.plot(range(len(valid_loss_his)),valid_loss_his,label="valid_curve")
    plt.legend()
    plt.savefig("plot/loss_lc/"+opt.wp+'.png', bbox_inches='tight', pad_inches=0)
def phase_eventGT_prediction_plot(model, test_data,opt):
    GT_history=[]
    p1_pred_history=[]
    p2_pred_history=[]
    p3_pred_history=[]


    model_path="checkpoint/tau/"+opt.gene+'_'+opt.method+'_'+opt.imp+'_'+str(opt.epoch)+"_"+str(opt.time_step)
    
    model.load_state_dict(torch.load(opt.wp+"phase1.pth"))
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            #event_time = event_time.to(torch.float)
            train_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            """ forward """
            output, prediction, enc_out = model(train_input,test_target)
            
            GT_history = np.append(GT_history,test_target.cpu())            
            p1_pred_history = np.append(p1_pred_history,prediction.cpu())
    
    model.load_state_dict(torch.load(opt.wp+"phase2.pth"))
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            #event_time = event_time.to(torch.float)
            train_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            """ forward """
            output, prediction, enc_out = model(train_input,test_target)
            
            p2_pred_history = np.append(p2_pred_history,prediction.cpu())

    model.load_state_dict(torch.load(opt.wp+"phase3.pth"))
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            #event_time = event_time.to(torch.float)
            train_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            """ forward """
            output, prediction, enc_out = model(train_input,test_target)
            p3_pred_history = np.append(p3_pred_history,prediction.cpu())
    p1_gosa=abs(p1_pred_history-GT_history)
    p2_gosa=abs(p2_pred_history-GT_history)
    p3_gosa=abs(p3_pred_history-GT_history)
    
    len=p1_gosa.shape[0]
    hako=np.array([0.25,0.5,0.75])
    p1_gosa.sort()
    p2_gosa.sort()
    p3_gosa.sort()
    print(p1_gosa[(len*hako).astype(int)])
    print(p2_gosa[(len*hako).astype(int)])
    print(p3_gosa[(len*hako).astype(int)])

    plt.clf()
    plt.figure(figsize=(20,4))
    plt.xlabel("event iD",fontsize=18)
    plt.ylabel("elapsed time",fontsize=18)
    
    plt.plot(range(300),GT_history[200:500],label="ground-truth")
    plt.plot(range(300),p1_pred_history[200:500],label="p1pred",linestyle="dotted")
    plt.plot(range(300),p2_pred_history[200:500],label="p2pred",linestyle="dashed")
    plt.plot(range(300),p3_pred_history[200:500],label="p3pred",linestyle="dashdot")
    
    plt.legend(fontsize=18, loc='upper right')
    plt.savefig("./plot/jisin_GT_pred/ID_time"+opt.imp+".svg", bbox_inches='tight', pad_inches=0)
    plt.savefig("./plot/jisin_GT_pred/ID_time"+opt.imp+".png", bbox_inches='tight', pad_inches=0)
