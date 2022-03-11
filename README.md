# MSCTD
Data and code for the ACL2022 main conference paper [MSCTD: A Multimodal Sentiment Chat Translation Dataset](https://arxiv.org/abs/2202.13645).

# Introduction
Still updating........................

# Training (Taking Zh->En as an example)
Our code is basically based on the publicly available toolkit: [THUMT-Tensorflow](https://github.com/THUNLP-MT/THUMT) (our python version 3.6).
The following steps are training our model and then test its performance in terms of BLEU, METEOR, and TER.

## Only text-based models

+ Training M1 (Trans.) and test, [checkpoint](), and [output]()

```
1) bash train_zh2en_share_base.sh # Suppose the generated checkpoint file is located in M1_path1
2) bash test_zhen_m1.sh checkpoint_name checkpoint_step test
```

+ Training M2 (TCT) and test, [checkpoint](), and [output]()

```
1) bash pretrain_train_tct.sh 1 1 # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
2) python thumt1_code/thumt/scripts/combine_add.py --model path2 --part M1_path1 --output path3  # copy the weight of the first stage to the second stage.
3) bash pretrain_train_tct.sh start_steps training_step # Here, set the --output=path3 and set the start_steps = first_stage_step and the training_step=first_stage_step + 10,000
4) bash test_zhen_tct.sh False False False checkpoint_name checkpoint_step test
```

+ Training M3 (CA-TCT) and test, [checkpoint](), and [output]()

```
1) bash pretrain_train_ca_tct.sh 1 1 # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
2) python thumt1_code/thumt/scripts/combine_add.py --model path2 --part M1_path1 --output path3  # copy the weight of the first stage to the second stage.
3) bash pretrain_train_ca_tct.sh start_steps training_step # Here, set the --output=path3 and set the start_steps = first_stage_step and set the training_step=first_stage_step + 10,000
4) bash test_zhen_tct.sh True True True checkpoint_name checkpoint_step test
```

## (T+CSV)-based models
+ Training M5 (Trans.+Sum) and test, [checkpoint](), and [output]()

```
1) bash train_multimodal_coarse_sum.sh 
2) bash test_zhen_coarse_sum.sh checkpoint_name checkpoint_step test # note that when testing you should set the "trainable=True" in Line206 of src_code/c-thumt-sum/thumt/models/transformer.py, to load the image feature.
```

+ Training M6 (Trans.+Att) and test, [checkpoint](), and [output]()

```
1) bash train_multimodal_coarse_att.sh  # Suppose the generated checkpoint file is located in M6_path1
2) bash test_zhen_coarse_att.sh checkpoint_name checkpoint_step test # note that when testing you should set the "trainable=True" in Line206 of src_code/c-thumt-sum/thumt/models/transformer.py, to load the image feature.
```
+ Training M7 (MCT) and test, [checkpoint](), and [output]()

```
1) bash pretrain_coarse_mct.sh 1 1 # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
2) python thumt1_code/thumt/scripts/combine_add.py --model path2 --part M6_path1 --output path3  # copy the weight of the first stage to the second stage.
3) bash pretrain_coarse_mct.sh # Here, set the --output=path3 and the training_step=first_stage_step + 5,000; Suppose the generated checkpoint file is path4
4) bash test_zhen_coarse_mct.sh False False False checkpoint_name checkpoint_step test  # note that when testing you should set the "trainable=True" in Line407 of src_code/thumt-dialog-wo-sp-decoder-w-mask-all-mlp-four-coarse-nct-att2/thumt/models/transformer.py, to load the image feature.
```

+ Training M8 (CA-MCT) and test, [checkpoint](), and [output]()

```
1) bash pretrain_coarse_ca_mct.sh 1 1 # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
2) python thumt1_code/thumt/scripts/combine_add.py --model path2 --part M6_path1 --output path3  # copy the weight of the first stage to the second stage.
3) bash pretrain_coarse_ca_mct.sh # Here, set the --output=path3 and the training_step=first_stage_step + 5,000; Suppose the generated checkpoint file is path4
4) bash test_zhen_coarse_mct.sh True True True checkpoint_name checkpoint_step test  # note that when testing you should set the "trainable=True" in Line407 of src_code/thumt-dialog-wo-sp-decoder-w-mask-all-mlp-four-coarse-nct-att2/thumt/models/transformer.py, to load the image feature.
```

## (T+FOV)-based models
+ Training M9 (Trans.+Con) and test, [checkpoint](), and [output]()

```
1) bash train_multimodal_fine_con_new.sh 
2) bash test_zhen_fine_con.sh.sh checkpoint_name checkpoint_step test 
```
+ Training M11 (M-Trans.) and test, [checkpoint](), and [output]()

```
1) bash train_multimodal_fine_m.sh   # Suppose the generated checkpoint file is located in M11_path1
2) bash test_zhen_fine_m.sh checkpoint_name checkpoint_step test 
```
+ Training M12 (MCT) and test, [checkpoint](), and [output]()

```
1) bash pretrain_train_mct.sh 1 1 # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
2) python thumt1_code/thumt/scripts/combine_add.py --model path2 --part M11_path1 --output path3  # copy the weight of the first stage to the second stage.
3) bash pretrain_train_mct.sh # Here, set the --output=path3 and the training_step=first_stage_step + 5,000; Suppose the generated checkpoint file is path4
4) bash test_zhen_fine_mct.sh False False False checkpoint_name checkpoint_step test  
```

+ Training M13 (CA-MCT) and test, [checkpoint](), and [output]()

```
1) bash pretrain_train_ca_mct.sh 1 1 # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
2) python thumt1_code/thumt/scripts/combine_add.py --model path2 --part M11_path1 --output path3  # copy the weight of the first stage to the second stage.
3) bash pretrain_train_ca_mct.sh # Here, set the --output=path3 and the training_step=first_stage_step + 5,000; Suppose the generated checkpoint file is path4
4) bash test_zhen_fine_mct.sh True True True checkpoint_name checkpoint_step test  
```

## Test by SacreBLEU, TER, and Meteor
Required TER: v0.7.25; Sacre-BLEU: version.1.4.13 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.13)

```
6) python SacreBLEU_TER_Coherence_Evaluation_code/cal_bleu_ter4ende.py # Please correctly set the golden file and predicted file in this file and in sacrebleu_ende.py, respectively.
```
+ Test by meteor (we used meteor-1.5.)

```
7) java -Xmx2G -jar meteor-*.jar generated_file reference_file -norm -writeAlignments -f system1
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

Please feel free to open an issue or email me (yunlonliang@gmail.com) for questions and suggestions.
