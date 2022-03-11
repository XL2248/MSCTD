# MSCTD Raw Data

All English utterances in MSCTD are from [OpenViDial](https://github.com/ShannonAI/OpenViDial), which may have been corrected or modified by human. When automatic annotation for enzh, the used English-Chinese subtitles are crawled from [kexiaoguo](https://www.kexiaoguo.com/), which are aligned by several tools, i.e., [Vecalign](https://github.com/thompsonb/vecalign) and [LASER](https://github.com/facebookresearch/LASER) (more details can be found in our paper). Now, the final English-Chinese subtitle database can be downloaded [here](https://drive.google.com/drive/folders/1Hf4Bs_nh3xN-1wzZk8eahWzJ8CLdDqDd?usp=sharing).

## EnZh
The main folder `enzh` contains training/dev/test sets, each of which is made up by the following files ('*' means dev/test/train): 
```
├──enzh
      └── english_*.txt // each line corresponds to an English dialogue text utterence.
      └── chinese_*.txt // each line corresponds to a Chinese dialogue text utterence, which is paired with the corresponding English utterance.
      └── chinese_*_seg.txt // each line corresponds to a segmented Chinese dialogue text utterence by Stanford CoreNLP toolkit.
      └── img_index_*.txt // each line is an episode of dialogue, which is a list of IDs (staring with 0).     
      └── sentiment_*.txt //positive: 2, negative: 1, and neutral:0
```
The enzh images can be downloaded here (saved by baiduyun(code：mlwe)/google driver). [enzh_train](https://pan.baidu.com/s/1e9jucSaq0i8uBPEvR_F5LQ) [test](https://drive.google.com/file/d/1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W/view?usp=sharing) [dev](https://drive.google.com/file/d/12HM8uVNjFg-HRZ15ADue4oLGFAYQwvTA/view?usp=sharing)

```
├── *_images // containing images (visual contexts) in which the text utterence takes place, with ID being the image filename (0,1,2, etc)
      └── 0.jpg
      └── 1.jpg
      └── ...
```
## EnDe
The main folder `ende` contains training/dev/test sets, each of which is made up by the following files ('*' means dev/test/train):
```
├──ende
      └── english_*.txt // each line corresponds to an English dialogue text utterence.
      └── german_*.txt // each line corresponds to a German dialogue text utterence, which is paired with the corresponding English utterance.
      └── img_index_*.txt // each line is an episode of dialogue, which is a list of IDs (staring with 0).     
      └── sentiment_*.txt //positive: 2, negative: 1, and neutral:0
```
The ende images can be downloaded here (Note that the test and valid sets are the same file for enzh and ende). [ende_train](https://drive.google.com/file/d/1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj/view?usp=sharing) [test](https://drive.google.com/file/d/1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W/view?usp=sharing) [dev](https://drive.google.com/file/d/12HM8uVNjFg-HRZ15ADue4oLGFAYQwvTA/view?usp=sharing)
```
├── *_images // containing images (visual contexts) in which the text utterence takes place, with ID being the image filename (0,1,2, etc)
      └── 0.jpg
      └── 1.jpg
      └── ...
```
