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
typ=sys.argv[1]
group_ids = load_dialogue_ids('../bpe_ende/image_index_%s.txt'%typ)
#with open():
#typ="test"english_train.txt.norm.tok
f4="../bpe_ende/english_%s.txt.norm.tok"%typ
f9="../bpe_ende/german_%s.txt.norm.tok"%typ
f_4="%s_en.ref"%typ
f_9="%s_de.ref"%typ
with open(f4, 'r', encoding='utf-8') as fr4, open(f9, 'r', encoding='utf-8') as fr9, open(f_4, 'w', encoding='utf-8') as fw4, open(f_9, 'w', encoding='utf-8') as fw9:
    c4 = fr4.readlines()
    c9 = fr9.readlines()
    k = 0
    print(len(c9))
    for d_id, dialog in enumerate(group_ids):
        for u_id, item in enumerate(dialog):
            if u_id % 2 == 0:
                fw4.write(c4[k])
                fw9.write(c9[k])
            k += 1
