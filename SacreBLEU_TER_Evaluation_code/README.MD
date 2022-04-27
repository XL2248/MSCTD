
## Test by SacreBLEU, TER, and Meteor
+ Test by SacreBLEU and TER (Required TER: v0.7.25; Sacre-BLEU: version.1.4.13 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.13))

```
1) python SacreBLEU_TER_Coherence_Evaluation_code/cal_bleu_ter4ende.py # Please correctly set the golden file and predicted file in this file and in sacrebleu_ende.py, respectively.
```
+ Test by meteor for English or German(we used meteor-1.5.) 

```
2) java -Xmx2G -jar meteor-*.jar generated_file reference_file -norm -writeAlignments -f system1
```
+ Test by meteor for Chinese (we used meteor-1.5.). 

```
3) python SacreBLEU_TER_Coherence_Evaluation_code/cal_meteor_score_enzh.py path_of_generated_file start_step end_step # Note that we first transform character-level Chinese to special English word (e.g., word1, word2, ... refer to SacreBLEU_TER_Coherence_Evaluation_code/transform_to_enchar.py) and then we utilize the above command (2) to calculate meteor.
```
