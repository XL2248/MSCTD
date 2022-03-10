# MSCTD
Data and codes for the ACL2022 main conference paper [MSCTD: A Multimodal Sentiment Chat Translation Dataset](https://arxiv.org/abs/2202.13645).

# Introduction
updating..

# Training (Taking En->De as an example)
Our code is basically based on the publicly available toolkit: [THUMT-Tensorflow](https://github.com/THUNLP-MT/THUMT) (our python version 3.6).
The following steps are training our model and then test its performance in terms of BLEU, TER, and Sentence Similarity.

## Data Preprocessing
Please refer to the "data_preprocess_code" file.

## Two-stage Training

+ The first stage

```
1) bash train_ende_base_stage1.sh # Suppose the generated checkpoint file is located in path1
```
+ The second stage (i.e., fine-tuning on the chat translation data)

```
2) bash train_ende_base_stage2.sh # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
3) python thumt_stage1_code/thumt/scripts/combine_add.py --model path2 --part path1 --output path3  # copy the weight of the first stage to the second stage.
4) bash train_ende_base_stage2.sh # Here, set the --output=path3 and the training_step=first_stage_step + 5,000; Suppose the generated checkpoint file is path4
```
+ Test by multi-blue.perl

```
5) bash test_ende_stage2.sh # set the checkpoint file path to path4 in this script. # Suppose the predicted file is located in path5 at checkpoint step xxxxx
```
+ Test by SacreBLEU and TER
Required TER: v0.7.25; Sacre-BLEU: version.1.4.13 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.13)

```
6) python SacreBLEU_TER_Coherence_Evaluation_code/cal_bleu_ter4ende.py # Please correctly set the golden file and predicted file in this file and in sacrebleu_ende.py, respectively.
```
+ Test by meteor

```
7) we used meteor-1.5. java -Xmx2G -jar meteor-*.jar generated_file reference_file -norm -writeAlignments -f system1
```
# Citation
If you find this project helps, please cite our paper :)

```
@article{liang2022msctd,
  title={MSCTD: A Multimodal Sentiment Chat Translation Dataset},
  author={Liang, Yunlong and Meng, Fandong and Xu, Jinan and Chen, Yufeng and Zhou, Jie},
  journal={arXiv preprint arXiv:2202.13645},
  year={2022}
}
```
