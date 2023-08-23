import torch
import gradio as gr
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import math
import readline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DAMO-NLP/seqGPT-560m', help="model name or local path to model folder")
    parser.add_argument('--share', action='store_true', help='gradio shared or not')

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

    examples = [
        ['分类', "The Moon's orbit around Earth has", 'pos，neg', 4, 1.0, 1.0, 2.0, 6],
        ['分类', "李老板卖鱼，卖了三十框鱼\t李老板赚翻了", 'entailment，contradiction', 4, 1.0, 1.0, 2.0, 6],
        ['抽取', "The smooth Borealis basin in the Northern Hemisphere covers 40%", '百分比，方向', 4, 1.0, 1.0, 2.0, 6],
        ['抽取', "童装红蜻蜓团体温州儿童用品有限公司是红蜻蜓团体旗下全资子公司，创立于2003年中温州和红蜻蜓的关系是什么？", '饰演，祖籍，毕业院校，创始人，首都，代言人，总部地点', 4, 1.0, 1.0, 2.0, 6],
    ]
    tasks = ['分类', '抽取']

    def generate(task, sent, labels, beam_size=4, temperature=1, top_p=1.0, repetition_penalty=2.0, extract_label_batch=6.0):
        sent = sent.strip()
        task = task.strip()
        labels = labels.strip().replace(',', '，')
        if task == '抽取':
            extract_label_batch = int(extract_label_batch)
            labels = labels.split('，')
            labels = np.array_split(labels, math.ceil(len(labels) / extract_label_batch))
            p = ['输入: {}\n{}: {}\n输出: {}'.format(sent, task, '，'.join(l), GEN_TOK) for l in labels]
        else:
            p = '输入: {}\n{}: {}\n输出: {}'.format(sent, task, labels, GEN_TOK)

        input_ids = tokenizer(p,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=1024)
        input_ids = input_ids.to(model.device)
        outputs = model.generate(**input_ids,
                                 num_beams=beam_size,
                                 do_sample=False,
                                 max_new_tokens=256,
                                 temperature=temperature,
                                 top_p=top_p,
                                 repetition_penalty=float(repetition_penalty))
        input_ids = input_ids.get('input_ids', input_ids)
        outputs = [outputs.tolist()[i][len(input_ids[i]):] for i in range(len(outputs))]
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return ''.join(response)

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.components.Dropdown(label="Task", choices=tasks),
            gr.components.Textbox(label="Text"),
            gr.components.Textbox(label="Labels"),
            gr.Slider(1, 10, value=4, step=1),
            gr.Slider(0.0, 1, value=1.0, step=0.05),
            gr.Slider(0.0, 1, value=1.0, step=0.05),
            gr.Slider(0.0, 10, value=1.0, step=0.05),
            gr.Slider(1, 10, value=6.0, step=1),
        ],
        outputs=gr.outputs.Textbox(label="Generated Text"),
        examples=examples
    )

    demo.launch(share=args.share)
