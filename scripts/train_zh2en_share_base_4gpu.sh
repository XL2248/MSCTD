code_dir=thumt
work_dir=/apdcephfs/share_47076/yunlonliang/chatnmt/multimodal
data_dir=/apdcephfs/share_47076/yunlonliang/multimodal_mt/
vocab_dir=$data_dir
train_data=/apdcephfs/share_47076/yunlonliang/multimodal_mt/
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
steps=2
layer=$1
uc=$2
rd=$3
model_name=zh2en_model_base_share_4gpu_${steps}0w_sent_layer${layer}_uc${uc}_rd$rd
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/$model_name \
  --input $train_data/chinese_train_zh_seg.txt.norm.bpe $train_data/english_train.txt.norm.tok.lowercase.bpe \
  --vocabulary $vocab_dir/vocab.enzh.chat.txt $vocab_dir/vocab.enzh.chat.en4.txt \
  --validation $data_dir/chinese_dev_zh_seg.txt.norm.bpe \
  --references $data_dir/english_dev.txt.norm.tok.lowercase \
  --parameters=device_list=[0,1,2,3],update_cycle=$uc,eval_steps=5000,train_steps=${steps}00000,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=$rd,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,shared_source_target_embedding=True,position_info_type="absolute",num_encoder_layers=$layer,num_decoder_layers=$layer

chmod 777 -R ${work_dir}/models/$model_name
