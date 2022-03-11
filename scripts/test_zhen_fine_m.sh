code_dir=fine-thumt-m
work_dir=/path/to/word_dir/src_code
data_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
vocab_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
train_data=/path/to/MSCTD/MSCTD_processed_data/zhen
image_data=/path/to/MSCTD/MSCTD_processed_data/object_features/zhen
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
data_name=test
checkpoint_dir=models_zhen/$1
# for alpha/beta
Step=$2
data_name=$3
#for idx in $Step
#for idx in `seq 200001 50 200500`
LOG_FILE=$work_dir/$checkpoint_dir/infer_${data_name}.log
output_dir=$work_dir/$checkpoint_dir
#for idx in `seq 55000 5000 300000`
for idx in $Step
do
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/$checkpoint_dir/checkpoint |tee -a ${LOG_FILE}
    cat $work_dir/$checkpoint_dir/checkpoint
    echo decoding with checkpoint-$idx |tee -a ${LOG_FILE}
    python $work_dir/$code_dir/thumt/bin/translator.py \
        --models transformer \
        --checkpoints $work_dir/$checkpoint_dir \
        --input $data_dir/${data_name}_zh.txt \
        --valid_index $data_dir/${data_name}_image_index \
        --valid_object $image_data/${data_name}.objects.mmap \
        --valid_object_mask $image_data/${data_name}.objects_mask.mmap \
        --vocabulary $vocab_dir/vocab.enzh.chat.txt $vocab_dir/vocab.enzh.chat.en4.txt $vocab_dir/all_image_idx.txt \
        --output $output_dir/"$data_name".out.en.$idx \
        --parameters=decode_batch_size=64,shared_source_target_embedding=True
    echo evaluating with checkpoint-$idx |tee -a ${LOG_FILE}
    sed -r "s/(@@ )|(@@ ?$)//g" $output_dir/"$data_name".out.en.$idx > $output_dir/${data_name}.out.en.delbpe.$idx
    echo "multi-bleu:" |tee -a ${LOG_FILE}
    $data_dir/multi-bleu.perl $data_dir/english_${data_name}.txt.norm.tok.lowercase.enzh < $output_dir/${data_name}.out.en.delbpe.$idx |tee -a ${LOG_FILE}
    echo finished of checkpoint-$idx |tee -a ${LOG_FILE}
done
