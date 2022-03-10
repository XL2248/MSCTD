#coding=utf-8

import os, sys
# for our base
checkpoint_step = [xxxxx]

for idx in checkpoint_step:
    os.system("python transform_xml.py tst ./ende_output/test English German ./ende_output/test.out.de.delbpe.%d"%idx) ## transform to *.xml format
    os.system("java -jar path_to/tercom-0.7.25/tercom.7.25.jar -r 'xml_data/ende_ref.xml' -h './ende_output/test_tst.xml' -N -s") # Golden file: 'xml_data/ende_ref.xml'; Predicted file: './ende_output/test_tst.xml'
    os.system("python sacrebleu_ende.py ./ende_output/test.out.de.delbpe.%d"%idx)
    print("Testing ende-checkpoint-%d"%idx)
