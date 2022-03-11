code_dir=thumt1_code
work_dir=/path/to/word_dir/src_code
data_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
vocab_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
train_data=/path/to/MSCTD/MSCTD_processed_data/zhen
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
steps=1
model_name=zh2en_model_base_share_${steps}0w_sent
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models_zhen/$model_name \
  --input $train_data/train_zh.txt $train_data/train_en.txt \
  --vocabulary $vocab_dir/vocab.enzh.chat.txt $vocab_dir/vocab.enzh.chat.en4.txt \
  --validation $data_dir/dev_zh.txt \
  --references $data_dir/english_dev.txt.norm.tok.lowercase.enzh \
  --parameters=device_list=[0,1,2,3],update_cycle=1,eval_steps=2000,train_steps=${steps}00000,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,shared_source_target_embedding=True,position_info_type="absolute"

chmod 777 -R ${work_dir}/models_zhen/$model_name
