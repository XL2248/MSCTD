#coding=utf-8
import os, sys
import code

word_num = 0
utt_num = 0 
d = {}
file=sys.argv[1]
with open(file, 'r', encoding='utf-8') as fr:
#for d_id, dialog in enumerate(group_ids):
    for item in fr.readlines():
#    dialog_num += 1
        utt_num += 1
        for w in item.strip().split():
            if w not in d.keys():
                d[w] = 1
            else:
                d[w] += 1
            word_num += 1
    # for u_id, item in enumerate(dialog):
print(utt_num, word_num, len(set(d)), d)
#code.interact(local=locals())
