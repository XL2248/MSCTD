import os,math
import csv
import xlsxwriter
import xlwt
from PIL import Image
import code
import json
import logging
import re
from typing import List
#from interval import Interval,IntervalSet
import os, sys
import shutil

def load_dialogue_ids(input_path, split="train") -> List[List[str]]:
    """load origin text data"""
    output = []
#    input_path = os.path.join(data_dir, f'{split}.dialogue.jsonl')
    logging.info(f"Loading origin data from {input_path}")
    with open(input_path, 'r', encoding='utf-8-sig') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sents = json.loads(line)
            output.append(sents)
#            code.interact(local=locals())
#            output.append([x.replace("\u2013", "-") for x in sents])  # todo delete after re-generating data
    logging.info(f"Loaded {sum(len(x) for x in output)} sentences from {input_path}")
    return output
typ="dev"
typ=sys.argv[1]
group_ids = load_dialogue_ids('../image_index_%s.txt'%typ)
#with open():
#typ="test"
f1="../bpe_zhen/%s_en_ctx_src.txt"%typ
f2="../bpe_zhen/%s_en_ctx.txt"%typ
f3="../bpe_zhen/%s_en_sample.txt"%typ
f4="../bpe_zhen/%s_en.txt"%typ
f5="../%s_image_idx.txt"%typ
f6="../bpe_zhen/%s_zh_ctx_src.txt"%typ
f7="../bpe_zhen/%s_zh_ctx.txt"%typ
f8="../bpe_zhen/%s_zh_sample.txt"%typ
f9="../bpe_zhen/%s_zh.txt"%typ
f_1="%s_en_ctx_src.txt"%typ
f_2="%s_en_ctx.txt"%typ
f_3="%s_en_sample.txt"%typ
f_4="%s_en.txt"%typ
f_5="%s_image_idx.txt"%typ
f_6="%s_zh_ctx_src.txt"%typ
f_7="%s_zh_ctx.txt"%typ
f_8="%s_zh_sample.txt"%typ
f_9="%s_zh.txt"%typ
with open(f1, 'r', encoding='utf-8') as fr1, open(f2, 'r', encoding='utf-8') as fr2, open(f3, 'r', encoding='utf-8') as fr3, open(f4, 'r', encoding='utf-8') as fr4, open(f5, 'r', encoding='utf-8') as fr5, open(f6, 'r', encoding='utf-8') as fr6, open(f7, 'r', encoding='utf-8') as fr7, open(f8, 'r', encoding='utf-8') as fr8, open(f9, 'r', encoding='utf-8') as fr9, open(f_1, 'w', encoding='utf-8') as fw1, open(f_2, 'w', encoding='utf-8') as fw2, open(f_3, 'w', encoding='utf-8') as fw3, open(f_4, 'w', encoding='utf-8') as fw4, open(f_5, 'w', encoding='utf-8') as fw5, open(f_6, 'w', encoding='utf-8') as fw6, open(f_7, 'w', encoding='utf-8') as fw7, open(f_8, 'w', encoding='utf-8') as fw8, open(f_9, 'w', encoding='utf-8') as fw9:
    c1 = fr1.readlines()
    c2 = fr2.readlines()
    c3 = fr3.readlines()
    c4 = fr4.readlines()
    c5 = fr5.readlines()
    c6 = fr6.readlines()
    c7 = fr7.readlines()
    c8 = fr8.readlines()
    c9 = fr9.readlines()
    k = 0
    for d_id, dialog in enumerate(group_ids):
        for u_id, item in enumerate(dialog):
            if u_id % 2 == 1: # change 1 to 0 for enzh or zhen.
                fw1.write(c1[k]) 
                fw2.write(c2[k])
                fw3.write(c3[k])
                fw4.write(c4[k])
                fw5.write(c5[k])
                fw6.write(c6[k])
                fw7.write(c7[k])
                fw8.write(c8[k])
                fw9.write(c9[k])
            k += 1
