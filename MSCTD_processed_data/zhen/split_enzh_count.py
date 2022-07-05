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
group_ids = load_dialogue_ids('../image_idx_%s.txt'%typ)
#with open():
#typ="test"
typ=sys.argv[2]
with open(typ, 'r', encoding='utf-8') as fr1, open(typ + ".enzh", 'w', encoding='utf-8') as fw1:
    c1 = fr1.readlines()
    k = 0
    for d_id, dialog in enumerate(group_ids):
        for u_id, item in enumerate(dialog):
            if u_id % 2 == 1:
                fw1.write(c1[k]) 
            k += 1
