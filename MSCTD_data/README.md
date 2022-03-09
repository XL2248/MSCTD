# MSCTD Data
The main folder `enzh` contains training/dev/test sets, each of which is made up by the following files ('*' means dev/test/train):
```
├──enzh
      └── english_*.txt // each line corresponds to a English dialogue text utterence.
      └── chinese_*.txt // each line corresponds to a Chinese dialogue text utterence, which is paired with the corresponding English utterance.
      └── chinese_*_seg.txt // each line corresponds to a segmented Chinese dialogue text utterence by [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/index.html).
      └── img_index_*.txt // each line is an episode of dialogue, which is a list of IDs (staring with 0).     
```
The enzh images can be downloaded here (Note that the test and valid is the same for enzh and ende). [enzh_train]() [test]() [valid]()

```
      └── *_images // containing images (visual contexts) in which the text utterence takes place, with ID being the image filename (0,1,2, etc)
            └── 0.jpg
            └── 1.jpg
            └── ...
```
The main folder `ende` contains training/dev/test sets, each of which is made up by the following files ('*' means dev/test/train):
```
├──ende
      └── english_*.txt // each line corresponds to a English dialogue text utterence.
      └── german_*.txt // each line corresponds to a German dialogue text utterence, which is paired with the corresponding English utterance.
      └── img_index_*.txt // each line is an episode of dialogue, which is a list of IDs (staring with 0).     
```
The ende images can be downloaded here. [ende_train]() [test]() [valid]()
```
      └── *_images // containing images (visual contexts) in which the text utterence takes place, with ID being the image filename (0,1,2, etc)
            └── 0.jpg
            └── 1.jpg
            └── ...
```