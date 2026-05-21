# [NeurIPS 2025] System Prompt Optimization with Meta-Learning
[![Paper](https://img.shields.io/badge/arXiv-2505.09666-b31b1b)](https://arxiv.org/abs/2505.09666)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange)](https://www.python.org/downloads/release/python-310s0/)
[![GCC](https://img.shields.io/badge/gcc-9.1%2B-blue)](https://gcc.gnu.org/gcc-9/)

🚀 **Welcome to the official repository of** [**System Prompt Optimization with Meta-Learning**](https://arxiv.org/abs/2505.09666)!

## 🔍 Overview
![MetaSPO](asset/main_fig.jpg)
This repository contains the official implementation of Meta-level System Prompt Optimizer (MetaSPO), a meta-learning approach for optimizing system prompts for Large Language Models (LLMs). MetaSPO is designed to optimize system prompts that are robust to diverse user inputs and transferable across a wide range of tasks and domains.

## 📌 Get Started
### Installation
```bash
git clone https://github.com/Dozi01/MetaSPO.git
cd MetaSPO
conda create -n metaspo python=3.10 -y
conda activate metaspo
pip install -r requirements.txt
```
Store your OpenAI credentials in the `.env` file:

```
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

`OPENAI_BASE_URL` is optional (defaults to the official OpenAI API). You can also pass `--openai_base_url` on the command line to override the env value.

### MetaSPO: Training and Evaluation
```bash
./main.sh
```
Refer to `main.sh` for detailed instructions.

### Tasks
Modify `configs/$DOMAIN.yaml` to set dataset configurations.  
To implement new tasks, include the task name in `srt/tasks/__init__.py` and implement a corresponding task class.

## 📜 Citation
If you find this work useful, please cite our paper:
```
@article{choi2025promptoptimizationmetalearning,
      title={System Prompt Optimization with Meta-Learning}, 
      author={Yumin Choi and Jinheon Baek and Sung Ju Hwang},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09666}, 
}
```
