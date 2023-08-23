import math
import torch
import argparse

import numpy as np

from time import sleep
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DAMO-NLP/seqGPT-560m', help="model name or local path to model folder")

    args = parser.parse_args()
    model_name_or_path = args.model
    print('Loading model: {}'.format(model_name_or_path))

    # We suggest to extract no more than N labels, if exceed N, split labels into chunks as smaller N results higher recall.
    default_extract_label_batch = 6

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    GEN_TOK = '[GEN]'

    # half and cuda, enforce padding and truncate at left
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if torch.cuda.is_available():
        model = model.half().cuda()

    model.eval()
    while True:
        sent = input('输入/Input: ').strip()
        task = input('分类/classify press 1, 抽取/extract press 2: ').strip()
        labels = input('标签集/Label-Set (e.g, labelA,LabelB,LabelC): ').strip().replace(',', '，')
        task = '分类' if task == '1' else '抽取'

        if task == '抽取':
            extract_label_batch = input('Extract_label_batch (Press enter to use the default value): ').strip()
            if extract_label_batch: 
                extract_label_batch = int(extract_label_batch)
            else:
                extract_label_batch = default_extract_label_batch


            labels = labels.split('，')
            labels = np.array_split(labels, math.ceil(len(labels) / extract_label_batch))
            p = ['输入: {}\n{}: {}\n输出: {}'.format(sent, task, '，'.join(l), GEN_TOK) for l in labels]
            # print(p)
        else:
            p = '输入: {}\n{}: {}\n输出: {}'.format(sent, task, labels, GEN_TOK)

        input_ids = tokenizer(p,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=1024)
        input_ids = input_ids.to(model.device)
        outputs = model.generate(**input_ids,
                                    num_beams=4,
                                    do_sample=False,
                                    max_new_tokens=256,
                                    temperature=1.0,
                                    top_p=1.0,
                                    repetition_penalty=2.0)
        input_ids = input_ids.get('input_ids', input_ids)
        outputs = [outputs.tolist()[i][len(input_ids[i]):] for i in range(len(outputs))]
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print('BOT: ========== ')
        print(''.join(response))
        sleep(1)