code_dir=thumt-dialog-wo-sp-decoder-w-mask-all-mlp-four-fine
work_dir=/apdcephfs/share_47076/yunlonliang/chatnmt/multimodal
data_dir=/apdcephfs/share_47076/yunlonliang/multimodal_mt/zhen
image_data=/apdcephfs/share_47076/yunlonliang/multimodal_mt/object_features/zhen
train_data=$data_dir
vocab_dir=/apdcephfs/share_47076/yunlonliang/multimodal_mt
kl_steps1=5000
kl_steps2=5000
crg=$1
mrg=$2 #True #True
coh=$3
clus=False 
cluts=False
clm=False
start_steps=$4
train_steps=$5
kl_steps2=${6}000
kl_steps1=$kl_steps2
rd=$7
num=4
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=direct_train_NCT_zh2en_crg${crg}_mrg${mrg}_coh${coh}_clus${clus}
model_name=pretrain_zh2en_crg${crg}_mrg${mrg}_coh${coh}_clus${clus}
model_name=pretrain_zh2en_base_uc1_crg${crg}_mrg${mrg}_coh${coh}
model_name=pretrain_zh2en_TCT_uc1_crg${crg}_mrg${mrg}_coh${coh}_dp$rd
#model_name=debug_test_pretrain_fine1
#model_name=init_model/init4zh2en_fine_4layer_uc1_crg${crg}_mrg${mrg}_coh${coh}
model_name=pretrain_zh2en_fine_TCT_uc1_crg${crg}_mrg${mrg}_coh${coh}_dp$rd
perfix=.nodup.clean
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models_zhen/$model_name \
  --input $train_data/train_zh.txt $train_data/train_en.txt $image_data/train.objects.mmap $image_data/train.objects_mask.mmap \
  --vocabulary $vocab_dir/vocab.enzh.chat30k.txt $vocab_dir/vocab.enzh.chat.en4.txt $vocab_dir/position.txt \
  --validation $data_dir/dev_zh.txt \
  --references $data_dir/english_dev.txt.norm.tok.lowercase.enzh \
  --context_source $train_data/train_zh_ctx_src.txt \
  --dialog_src_context $train_data/train_zh_ctx.txt \
  --dialog_tgt_context $train_data/train_en_ctx.txt \
  --sample $train_data/train_en_sample.txt \
  --dev_context_source $data_dir/dev_zh_ctx_src.txt \
  --dev_dialog_src_context $data_dir/dev_zh_ctx.txt \
  --dev_dialog_tgt_context $data_dir/dev_en_ctx.txt \
  --dev_sample $data_dir/dev_en_sample.txt \
  --parameters=device_list=[0,1,2,3],update_cycle=1,eval_steps=2000,train_steps=$train_steps,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=$rd,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,shared_source_target_embedding=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus,use_cluts=$cluts,use_clm=$clm,start_steps=$start_steps,position_info_type="absolute",num_encoder_layers=$num,num_decoder_layers=$num,image_num=62593,dev_object=$image_data/dev.objects.mmap,dev_object_mask=$image_data/dev.objects_mask.mmap

chmod 777 -R ${work_dir}/models_zhen/$model_name
