# Z1-Coder Training

### ⚙️ Setup

We recommend using [CONDA](https://docs.conda.io/projects/miniconda) to manage your environment. Run the following commands to setup your environment:

```sh
conda create -n z1train python=3.9
conda activate z1train
cd src/train
pip install -r requirements.txt
```

### ⚡️ Training
Our training scripts refer to [Fastchat](https://github.com/lm-sys/FastChat). To train a model, run the following command:

```sh
cd src
bash train/script/train_qwen.sh
```