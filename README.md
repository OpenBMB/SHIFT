<div align="center">
<h1> SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict Mitigation in Retrieval-Augmented Generation
<h5 align="center"> 
  
<a href='xxx'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/Qwen/Qwen3-32B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-模型的名字-blue'>
<a href='https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-模型的名字-blue'>


Ruochang Li<sup>1*</sup>,
Pengcheng Huang<sup>1\*</sup>,
Zhenghao Liu<sup>1†</sup>,
Yukun Yan<sup>2</sup>,
</br>
Huiyuan Xie<sup>2</sup>,
Yu Gu<sup>1</sup>,
Ge Yu<sup>1</sup>,
Maosong Sun<sup>2</sup>

<sup>1</sup>Northeastern University, <sup>2</sup>Tsinghua University

</h5>
</div>


## 📖 Overview
<p align="center">
  <img src="figs/framework.png" alt="SHIFT overview" width="95%">
</p>



## ⚙️ Setup
```bash
conda create --name shift python==3.10.0
conda activate shift
git clone https://github.com/NEUIR/SHIFT.git
cd SHIFT
```
1. Install PyTorch
```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
2. Install Flash Attention
```
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
3. Install the rest of the dependencies:
```
pip install -r requirements.txt
```
4. Patch vLLM model files  
This project requires modifications to the vLLM implementations of Qwen3 and LLaMA.  
After installing vLLM, replace the corresponding files in your current Python environment:
```
VLLM_DIR=$(python -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)")

cp SHIFT/vllm/qwen3.py "$VLLM_DIR/model_executor/models/qwen3.py"
cp SHIFT/vllm/llama.py "$VLLM_DIR/model_executor/models/llama.py"
```
You can verify the patched files with:
```
python -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)"
```
The expected target paths look like:
```
/path/to/your/env/lib/python3.10/site-packages/vllm/model_executor/models/qwen3.py
/path/to/your/env/lib/python3.10/site-packages/vllm/model_executor/models/llama.py
```
Note: the vLLM patch should be applied after running ```pip install -r requirements.txt```, because installing or reinstalling vLLM may overwrite these files.


### Data

Our corresponding generated training data is placed under the dataset folder.

Download the files from [MRQA-Shared-Task-2019](https://github.com/mrqa/MRQA-Shared-Task-2019).  
Use the downloaded data to synthesize the data using [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG).


### Training
GRPO with a single GPU: 
```
python single_gpu.py
```
GRPO with multiple GPUs:
```
python multi_gpu.py
```


### Evaluation
For MRQA and ConfiQA: 
```
python eval.py
```
For MMLU, use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)


### Analysis
We also provide the t-SNE visualization pipeline for gates in SHIFT, with corresponding figures available under the figs folder:
```
python run_batch_tsne.py
```

For Qwen-3-0.6B: 
<p align="center">
  <img src="figs/Qwen-3-0.6B-tsne.png" alt=" Qwen-3-0.6B" width="50%">
</p>
For Qwen-3-8B:
<p align="center">
  <img src="figs/Qwen-3-8B-tsne.png" alt=" Qwen-3-8B" width="50%">
</p>



## 📄 Acknowledgement 
Our work is built on the following codebases, and we are deeply grateful for their contributions.
- [vllm](https://github.com/vllm-project/vllm)
- [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [gated_attention](https://github.com/qiuzh20/gated_attention)
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
  


## 🥰 Citation
If you find this work useful, please cite our paper and give us a shining star 🌟
```bibtex
@article{Li2026shift,
      title={SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict Mitigation in Retrieval-Augmented Generation},
      author={Li, Ruochang and Huang, Pengcheng and Liu, Zhenghao and Yan, Yukun and Xie, Huiyuan and Gu, Yu and Yu, Ge and Sun, Maosong},
      year={2026}
      url={}, 
}
```


## 📧 Contact
If you have questions or collaboration opportunities, please feel free to email:
```
ruochangli@gmail.com
```

