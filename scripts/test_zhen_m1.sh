code_dir=thumt1_code
work_dir=/path/to/word_dir/src_code
data_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
vocab_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
train_data=/path/to/MSCTD/MSCTD_processed_data/zhen

export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
data_name=test
checkpoint_dir=models_zhen/$1 # checkpoint name
# for alpha/beta
Step=$2 # checkpoint step
data_name=$3 # test/dev
LOG_FILE=$work_dir/$checkpoint_dir/infer_${data_name}.log
output_dir=$work_dir/$checkpoint_dir
for idx in $Step
do
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/$checkpoint_dir/checkpoint |tee -a ${LOG_FILE}
    cat $work_dir/$checkpoint_dir/checkpoint
    echo decoding with checkpoint-$idx |tee -a ${LOG_FILE}
    python $work_dir/$code_dir/thumt/bin/translator.py \
        --models transformer \
        --checkpoints $work_dir/$checkpoint_dir \
        --input $data_dir/${data_name}_zh.txt \
        --vocabulary $vocab_dir/vocab.enzh.chat.txt $vocab_dir/vocab.enzh.chat.en4.txt \
        --output $output_dir/"$data_name".out.en.$idx \
        --parameters=decode_batch_size=64,shared_source_target_embedding=True
    echo evaluating with checkpoint-$idx |tee -a ${LOG_FILE}
#    cd $train_dir
    sed -r "s/(@@ )|(@@ ?$)//g" $output_dir/"$data_name".out.en.$idx > $output_dir/${data_name}.out.en.delbpe.$idx
    echo "multi-bleu:" |tee -a ${LOG_FILE}
    $data_dir/multi-bleu.perl $data_dir/english_${data_name}.txt.norm.tok.lowercase.enzh < $output_dir/${data_name}.out.en.delbpe.$idx |tee -a ${LOG_FILE}
    echo finished of checkpoint-$idx |tee -a ${LOG_FILE}
done
