code_dir=thumt-dialog-wo-sp-decoder-w-mask-all-mlp-four
work_dir=/path/to/word_dir/src_code
data_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
vocab_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
train_data=/path/to/MSCTD/MSCTD_processed_data/zhen
kl_steps1=5000
kl_steps2=5000
crg=True
mrg=True #True 
coh=True #True
clus=False 
cluts=False
clm=False
start_steps=$1
train_steps=$2
kl_steps2=30000
kl_steps1=$kl_steps2
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=pretrain_zh2en_TCT_uc1_crg${crg}_mrg${mrg}_coh${coh}
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models_zhen/$model_name \
  --input $train_data/train_zh.txt $train_data/train_en.txt \
  --vocabulary $vocab_dir/vocab.enzh.chat.txt $vocab_dir/vocab.enzh.chat.en4.txt $vocab_dir/position.txt \
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
  --parameters=device_list=[0,1,2,3],update_cycle=1,eval_steps=1000,train_steps=$train_steps,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.4,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,shared_source_target_embedding=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus,use_cluts=$cluts,use_clm=$clm,start_steps=$start_steps,position_info_type="absolute",num_encoder_layers=6,num_decoder_layers=6

chmod 777 -R ${work_dir}/models_zhen/$model_name
