code_dir=c-thumt-sum
work_dir=/path/to/word_dir/src_code
data_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
vocab_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
train_data=/path/to/MSCTD/MSCTD_processed_data/zhen
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
steps=1
layer=4
uc=4
model_name=zh2en_model_coarse_sum_1gpu_${steps}0w_sent_layer${layer}_uc${uc}
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models_zhen/$model_name \
  --input $train_data/train_zh.txt $train_data/train_en.txt $train_data/train_image_idx.txt \
  --vocabulary $vocab_dir/vocab.enzh.chat.txt $vocab_dir/vocab.enzh.chat.en4.txt $vocab_dir/all_image_idx.txt \
  --validation $data_dir/dev_zh.txt \
  --valid_index $data_dir/dev_image_idx.txt \
  --references $data_dir/english_dev.txt.norm.tok.lowercase.enzh \
  --parameters=device_list=[0],update_cycle=$uc,eval_steps=2000,train_steps=${steps}00000,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.2,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,shared_source_target_embedding=True,position_info_type="absolute",num_encoder_layers=$layer,num_decoder_layers=$layer

chmod 777 -R ${work_dir}/models_zhen/$model_name
