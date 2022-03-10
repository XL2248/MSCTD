# Corpus BLEU with arguments
# Run this file from CMD/Terminal
# Example Command: python3 compute-bleu-args.py test_file_name.txt mt_file_name.txt


import sys
import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='de')

target_test = "test.tok.de"  # Test file argument
target_pred = sys.argv[1]  # MTed file argument

# Open the test dataset human translation file and detokenize the references
refs = []

with open(target_test) as test:
    for line in test: 
        line = line.strip().split() 
        line = md.detokenize(line) 
        refs.append(line)
    
#print("Reference 1st sentence:", refs[0])

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU


# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open(target_pred) as pred:  
    for line in pred: 
        line = line.strip().split() 
        line = md.detokenize(line) 
        preds.append(line)

#print("MTed 1st sentence:", preds[0])    


# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds, refs)
print("sacreBLEU: ", bleu.score)
