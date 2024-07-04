
# AlignGPT Family: Support Various Large Language Models and Visual Backbones

[[Project Page](https://aligngpt-vl.github.io/)] [[Paper](https://arxiv.org/abs/2405.14129)] [[Demo](http://47.116.173.89:7870/)] [[Model](https://huggingface.co/nlpzhaof)]


Authors: [Fei Zhao*](https://scholar.google.com/citations?user=V01xzWQAAAAJ&hl=zh-CN), Taotian Pang*, Chunhui Li, [Zhen Wu](https://scholar.google.com/citations?user=IoGlgtoAAAAJ&hl=zh-CN), Junjie Guo, Shangyu Xing, [Xinyu Dai](https://scholar.google.com/citations?user=zpWB1CgAAAAJ&hl=zh-CN)


<!-- ![architecture](./assert/architecture.png) -->

## News and Updates
- [7/03] 🔥 Our code supports a variety of large language models, including LLaMA-3-8B-Base, LLaMA-3-8B-Instruct, phi-1_5, phi-2, gemma-7b and Mistral-7B-Instruct-v0.2.
- [7/03] 🔥 Our code support a visual backbone siglip-so400m-patch14-384 and various image resolutions.

## Contents
- [Install](#install)
- [Model Zoo](#model-zoo)
- [Training](#training)
- [Evaluation](#evaluation)

## Install

### Docker

We recommend to use docker to prepare the environment.

1. Clone this repository and navigate to AlignGPT_family folder

```bash
git clone https://github.com/AlignGPT-VL/AlignGPT_family.git
cd AlignGPT_family
```

2. Build the docker image

```bash
cd deploy
docker build -t aligngpt:2.0 .
```

If your machine cannot connect to github to download the flash attention pip wheel, you can download it manually on https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl and put it to `deploy/flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`.

3. To start the container, run the following command in the project root directory

```bash
docker run --gpus all --ipc=host --network=host --rm -it -v .:/workspace aligngpt:2.0
```

More `-v` options can be added to mount the data and output directories.

### Conda

1. Clone this repository and navigate to AlignGPT_family folder

```bash
git clone https://github.com/AlignGPT-VL/AlignGPT_family.git
cd AlignGPT_family
```

2. Install Package

```Shell
conda create -n aligngpt python=3.10 -y
conda activate aligngpt
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r deploy/requirements.txt
```

Finally, you need to install flash-attention manually before running the model.

## Model Zoo

Please download the weights for LLM, Vision Backbone and place them in the `./playground/model` folder, we also provide all the weights for the AlignGPT checkpoint.

| Model | LLM | Vision Backbone | Image Resolution| Pre-training | Instruct-tuning | 
|----------|----------|-----------|---|---|---|
| AlignGPT | [LLaMA-3-8B-Base](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 336*336 | To be released|To be released|
| AlignGPT | [LLaMA-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 336*336 | To be released| To be released |
| AlignGPT | [phi-1_5](https://huggingface.co/microsoft/phi-1_5/tree/main) | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 336*336 | To be released|To be released|
| AlignGPT | [phi-2](https://huggingface.co/microsoft/phi-2/tree/main) | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 336*336 | To be released|To be released|
| AlignGPT | [gemma-7b](https://huggingface.co/google/gemma-7b/tree/main) | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 336*336 |To be released|To be released|
| AlignGPT | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main) | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 336*336 |To be released|To be released|
| AlignGPT | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384/tree/main) | 384*384 |To be released|To be released|

## Training

### Pre-training
* **Dataset**: We use the 558K image-text pairs in the pre-training phase. Organize them in `./playground/data` as follows:

```
├── LLaVA-Pretrain
│   └── blip_laion_cc_sbu_558k_with_similarity_number.json
│   └── images
```

* **Run**: You can launch the pre-training phase using the following command:
    ```
    bash scripts/pretrain.sh
    ```
Before running the script of pretraining, you should set the arguments related to **directories** of model checkpoints, data and outputs, *i.e.*, `model_name_or_path`, `data_path`, `image_folder`, `vision_tower` and `output_dir`.

### Instruction-tuning
* **Dataset**: We used 665K image-text pairs/text data in the instruction-tuning phase. The images corresponding to these data include: `COCO`, `GQA`, `OCR-VQA`, `TextVQA`, and `VisualGenome`. Organize them in `./playground/data` as follows:

```
├── llava_v1_5_mix665k.json
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

* **Run**: You can launch the instruction-tuning stage using the following command:
    ```
    bash scripts/finetune.sh
    ```
Before running the script of instruction tuning, you should set the argument `pretrain_mm_mlp_align`, which is the path where you store the weights of the pre-training phase.

## Evaluation

We conduct evaluation on 12 benchmarks. Here, we demonstrate how to evaluate the performance of our model on `MME` dataset. We use the following command to run the evaluation stage:
```
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh
```
You should set the directories of the model checkpoints and datasets in the scripts before running it. The evaluation of other datasets can be found in [Evaluation.md](docs/Evaluation.md).

## Citation
If you find AlignGPT useful for your research and applications, please cite using this BibTeX:
```
@misc{zhao2024aligngpt,
      title={AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability}, 
      author={Fei Zhao and Taotian Pang and Chunhui Li and Zhen Wu and Junjie Guo and Shangyu Xing and Xinyu Dai},
      year={2024},
      eprint={2405.14129},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement
We build our project based on [LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA).

## License

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
