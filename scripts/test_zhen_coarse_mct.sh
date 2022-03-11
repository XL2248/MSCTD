code_dir=thumt-dialog-wo-sp-decoder-w-mask-all-mlp-four-coarse-nct-att2
work_dir=/path/to/word_dir/src_code
data_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
vocab_dir=/path/to/MSCTD/MSCTD_processed_data/zhen
train_data=/path/to/MSCTD/MSCTD_processed_data/zhen
#vocab_dir=$train_data
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
data_name=test
crg=$1
mrg=$2 #True #True
coh=$3
clus=False
checkpoint_dir=models_zhen/$4
Step=$5
data_name=$6
output_dir=$work_dir/$checkpoint_dir
LOG_FILE=$output_dir/infer_${data_name}.log
for idx in $Step
do
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/$checkpoint_dir/checkpoint |tee -a ${LOG_FILE}
    cat $work_dir/$checkpoint_dir/checkpoint
    echo decoding with checkpoint-$idx |tee -a ${LOG_FILE}
    python $work_dir/$code_dir/thumt/bin/translator.py \
        --models transformer \
        --checkpoints $work_dir/$checkpoint_dir \
        --input $data_dir/${data_name}_zh.txt \
        --valid_index $data_dir/${data_name}_image_idx.txt \
        --vocabulary $vocab_dir/vocab.enzh.chat.txt $vocab_dir/vocab.enzh.chat.en4.txt $vocab_dir/position.txt $vocab_dir/all_image_idx.txt \
        --dev_context_source $data_dir/"$data_name"_zh_ctx_src.txt \
        --dev_dialog_src_context $data_dir/"$data_name"_zh_ctx.txt \
        --dev_dialog_tgt_context $data_dir/"$data_name"_en_ctx.txt \
        --dev_sample $data_dir/"$data_name"_en_sample.txt \
        --output $output_dir/"$data_name".out.en.$idx \
        --parameters=decode_batch_size=64,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus,shared_source_target_embedding=True
    echo evaluating with checkpoint-$idx |tee -a ${LOG_FILE}
    sed -r "s/(@@ )|(@@ ?$)//g" $output_dir/"$data_name".out.en.$idx > $output_dir/${data_name}.out.en.delbpe.$idx
    echo "multi-bleu:" |tee -a ${LOG_FILE}
    $data_dir/multi-bleu.perl $data_dir/english_${data_name}.txt.norm.tok.lowercase.enzh < $output_dir/${data_name}.out.en.delbpe.$idx |tee -a ${LOG_FILE}
    echo finished of checkpoint-$idx |tee -a ${LOG_FILE}
done
chmod 777 -R $output_dir
