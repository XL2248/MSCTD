code_dir=thumt-dialog-wo-sp-decoder-w-mask-all-mlp-four-fine
work_dir=/apdcephfs/share_47076/yunlonliang/chatnmt/multimodal
data_dir=/apdcephfs/share_47076/yunlonliang/multimodal_mt/zhen
vocab_dir=/apdcephfs/share_47076/yunlonliang/multimodal_mt
train_data=/apdcephfs/share_47076/yunlonliang/multimodal_mt/zhen
image_data=/apdcephfs/share_47076/yunlonliang/multimodal_mt/object_features/zhen
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
        --dev_context_source $data_dir/${data_name}_zh_ctx_src.txt \
        --dev_dialog_src_context $data_dir/${data_name}_zh_ctx.txt \
        --dev_dialog_tgt_context $data_dir/${data_name}_en_ctx.txt \
        --dev_sample $data_dir/${data_name}_en_sample.txt \
        --valid_index $data_dir/${data_name}_image_index \
        --valid_object $image_data/${data_name}.objects.mmap \
        --valid_object_mask $image_data/${data_name}.objects_mask.mmap \
        --vocabulary $vocab_dir/vocab.enzh.chat30k.txt $vocab_dir/vocab.enzh.chat.en4.txt $vocab_dir/position.txt \
        --output $output_dir/"$data_name".out.en.$idx \
        --parameters=decode_batch_size=64
    echo evaluating with checkpoint-$idx |tee -a ${LOG_FILE}
#    cd $train_dir
    sed -r "s/(@@ )|(@@ ?$)//g" $output_dir/"$data_name".out.en.$idx > $output_dir/${data_name}.out.en.delbpe.$idx
#    perl $data_dir/chi_char_segment.pl < $data_dir/${data_name}.out.zh.delbpe.$idx > $data_dir/${data_name}.out.zh.delbpe.char.$idx
#    $data_dir/chi_char_segment.pl < $data_dir/${data_name}.tok.zh > $data_dir/${data_name}.tok.zh.char
    #$data_dir/mteval-v11b.pl -s $data_dir/${data_name}_bpe.32k.en -r $data_dir/${data_name}.tok.ch.seg -t $data_dir/${data_name}.out.en.ch.delbpe.seg -c > $data_dir/exp.bleu
    #$data_dir/mteval-v11b.pl -s $data_dir/${data_name}_bpe.32k.en -r $data_dir/${data_name}.tok.ch.seg -t $data_dir/${data_name}.out.en.ch.delbpe.seg -c
    echo "multi-bleu:" |tee -a ${LOG_FILE}
    $data_dir/multi-bleu.perl $data_dir/english_${data_name}.txt.norm.tok.lowercase.enzh < $output_dir/${data_name}.out.en.delbpe.$idx |tee -a ${LOG_FILE}
    echo finished of checkpoint-$idx |tee -a ${LOG_FILE}
done
