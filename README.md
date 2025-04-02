# Installation

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

```bash
git clone https://github.com/PedroRASB/RadGPT
cd RadGPT
conda create -n vllm python=3.12 -y
conda activate vllm
conda install -y ipykernel
conda install -y pip
pip install vllm==0.6.1.post2
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install -r requirements.txt
mkdir HFCache
```

If you have problems with vllm, try running this before vllm serve commands:
```bash
export NCCL_P2P_DISABLE=1
```


# Generate Structured, Narrative and Fusion Reports

Use Rad-GPT to generate reports from organ and tumor per-voxel segmentations.

[RadGPTReportGeneration/README.md](RadGPTReportGeneration/README.md)

# Evaluation

LLM (labeler) extracts binary labels indicating if reports indicate the presence or absence of liver, kidney and pancreatic cancers. These labels can be used to compare AI-made reports to human-made reports (ground-truth) and evaluate cancer detection specificity and sensitivity.

[ReportEvaluationLLM/README.md](ReportEvaluationLLM/README.md)

