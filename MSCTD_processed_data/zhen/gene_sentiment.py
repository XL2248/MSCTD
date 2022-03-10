#coding=utf-8
import os, sys
import code

word_num = 0
utt_num = 0 
d = set()
file=sys.argv[1]
#n=sys.argv[2]
with open(file, 'r', encoding='utf-8') as fr, open(file + '.sentiment', 'w', encoding='utf-8') as fw:
#for d_id, dialog in enumerate(group_ids):
    con = fr.readlines()
    for idx in con:
        fw.write("sentiment" + idx.strip() + '\n')
    # for u_id, item in enumerate(dialog):
#code.interact(local=locals())
