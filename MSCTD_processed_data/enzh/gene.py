#coding=utf-8
import os, sys
import code

word_num = 0
utt_num = 0 
d = set()
file=sys.argv[1]
n=sys.argv[2]
with open(file, 'w', encoding='utf-8') as fr:
#for d_id, dialog in enumerate(group_ids):
    for idx in range(int(n)):
        fr.write(str(idx) + '\n')
    # for u_id, item in enumerate(dialog):
#code.interact(local=locals())
