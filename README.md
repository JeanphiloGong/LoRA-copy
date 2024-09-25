# LoRA-copy
LoRA is one of the famous method to fine-tune to Large Language Model, I create this repository to copy from the microsoft to learn this method, if you also like this method, welcome to talk with me.

# LoRA: Low-Rank Adaptation of Large Language Models

This repo contains the source code of the python package 'loralib' and several examples of how to integrate it with PyTorch models, such as those in Hugging Face.
We only support PyTorch for now.
See our paper for a detailed description of LoRA.

**LoRA: Low-Rank Adaptation of Large Language Models** <br>
*Edward J. Hu\*, Yelong Shen\*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* <br>
Paper: https://arxiv.org/abs/2106.09685 <br>
Video explainer: https://www.youtube.com/watch?v=DhRoTONcyZE <br>

*Update 2/2023: LoRA is now supported by the [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingfac/peft) library be Hugging Face.*

LoRA reduces the number of trainable parameters by learning pairs of rank-decompostion matrices while freezing the original weights.
This vastly reduces the requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment all without introducing inference latency.
LoRA also outperforms several other adaptation methods including adapter, prefix-tuning, and fine-tuning.

We obtain result comparable or superior to full finetuning on the GLUE benchmark using [RoBERTa (Liu et al., 2019)](https://arxiv.org/abs/1907.11692) and large and [DeBERTa (He et al., 20202)](https://arxiv.org/abs/2006.03654) XXL 1.5B, while only training and storing a fraction of the parameters. Click the numbers below to download the RoBERTa and DeBERTa LoRA checkpoints.

|   |         | RoBERTa base <br> Fine-tune  |  RoBERTa base <br> LoRA  | DeBERTa XXL <br> Fine-tune | DeBERTa XXL <br> LoRA  |
|---|-------------------------|----------------|--------------------------|-----------------|-----------------|
|   | # of Trainable Params.  | 125M | 0.8M | 1.5B | 4.7M     |
|   | MNLI (m-Acc/mm-Acc)     | <b>87.6</b> | [<b>87.5</b>±.3/86.9±.3](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_mnli.bin) |91.7/<b>91.9</b>| [<b>91.9</b>±.1/<b>91.9</b>±.2](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_mnli.bin)       |
|   | SST2 (Acc)              | 94.8 | [<b>95.1</b>±.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_sst2.bin) | <b>97.2</b>    | [96.9±.2](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_sst2.bin)                    |
|   | MRPC (Acc)              | <b>90.2</b> | [<b>89.7</b>±.7](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_mrpc.bin) | 92.0           | [<b>92.6</b>±.6](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_mrpc.bin)             |
|   | CoLA (Matthew's Corr)   | <b>63.6</b> | [<b>63.4</b>±1.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_cola.bin) | <b>72.0</b>    | [<b>72.4</b>±1.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_cola.bin)           |
|   | QNLI (Acc)              | 92.8 | [<b>93.3</b>±.3](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_qnli.bin) | <b>96.0</b>    | [<b>96.0</b>±.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_qnli.bin)            |
|   | QQP (Acc)               | <b>91.9</b> | [90.8±.1](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_qqp.bin) | 92.7           | [<b>92.9</b>±.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_qqp.bin)           |
|   | RTE (Acc)               | 78.7 | [<b>86.6</b>±.7](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_rte.bin) | 93.9           | [<b>94.9</b>±.4](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_rte.bin)           |
|   | STSB (Pearson/Spearman Corr) | 91.2 | [<b>91.5</b>±.2/<b>91.3</b>±.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_stsb.bin) |<b>92.9</b>/92.6| [<b>93.0</b>±.2/<b>92.9</b>±.3](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_stsb.bin)      |
|   | Average  | 86.40 | <b>87.24</b> | 91.06 | <b>91.32</b> |

<i>Note: You still need the original pre-trained checkpoint from [Hugging Face](https://huggingface.co/) to use the LoRA checkpoints.</i>

Fine-tuning numbers are taken from [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) and [He et al. (2020)](https://arxiv.org/abs/2006.03654).  We include confidence intervals on results from our experiments. Please follow the instructions in `examples/NLU/` to reproduce our results.

On GPT-2, LoRA compares favorably to both full finetuning and other efficient tuning methods, such as [adapter (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751) and [prefix tuning (Li and Liang, 2021)](https://arxiv.org/abs/2101.00190). We evaluated on E2E NLG Challenge, DART, and WebNLG:

|   | Method              | # of Trainable Params | E2E (BLEU)   | DART (BLEU)  | WebNLG (BLEU-U/S/A)            |
|---|---------------------|-----------------------|--------------|--------------|--------------------------------|
|   | GPT-2 M (Fine-Tune) | 354.92M               | 68.2         | 46.0         | 30.4/<b>63.2</b>/47.6          |
|   | GPT-2 M (Adapter)   | 0.37M                 | 66.3         | 42.4         | 45.1/54.5/50.2                 |
|   | GPT-2 M (Prefix)    | 0.35M                 | 69.7         | 45.7         | 44.1/63.1/54.4                 |
|   | GPT-2 M (LoRA)      | 0.35M                 |<b>70.4</b>±.1|<b>47.1</b>±.2| <b>46.7</b>±.4/62.1±.2/<b>55.3</b>±.2 |
|   | GPT-2 L (Fine-Tune) | 774.03M               | 68.5         | 46.5         | 41.7/<b>64.6</b>/54.2          |
|   | GPT-2 L (Adapter)   | 0.88M                 | 69.1±.1      | 45.7±.1      | <b>49.8</b>±.0/61.1±.0/56.0±.0 |
|   | GPT-2 L (Prefix)    | 0.77M                 | 70.3         | 46.5         | 47.0/64.2/56.4                 |
|   | GPT-2 L (LoRA)      | 0.77M                 |<b>70.4</b>±.1|<b>47.5</b>±.1| 48.4±.3/<b>64.0</b>±.3/<b>57.0</b>±.1 |

Non-LoRA baselines, except for adapter on GPT-2 large, are taken from [Li and Liang (2021)](https://arxiv.org/abs/2101.00190). We include confidence intervals on results from our experiments.

Download the GPT-2 LoRA checkpoints:
 * [GPT-2 Medium E2E](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_md_lora_e2e.pt) (1.5 MB)
 * [GPT-2 Medium DART](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_md_lora_dart.pt) (1.5 MB)
 * [GPT-2 Medium WebNLG](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_md_lora_webnlg.pt) (1.5 MB)
 * [GPT-2 Large E2E](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_lg_lora_e2e.pt) (2.3 MB)
 * [GPT-2 Large DART](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_lg_lora_dart.pt) (2.3 MB)
 * [GPT-2 Large WebNLG](https://github.com/microsoft/LoRA/releases/download/GPT-2/gpt2_lg_lora_webnlg.pt) (2.3 MB)

Please follow the instructions in `examples/NLG/` to reproduce our result.