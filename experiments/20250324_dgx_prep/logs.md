1. Move the dataset to the cluster
    ```bash
    rsync -ahzP /data_ssd/merged/standard_text_document.* ameyagod@dgx-login1:/lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B/
    ```
2. DGX container preparation (on a compute node)
    1. Get base container
        ```bash
        cd ~
        ln -s ln -s /lustre/fs0/scratch/ameyagod/images ~/images
        enroot import 'docker://$oauthtoken@nvcr.io#nvidia/pytorch:25.02-py3'
        enroot create --name gpt-neox nvidia+pytorch+25.02-py3.sqsh
        ```
    2. Install NeoX
        ```bash
        enroot start --rw --mount /lustre/fs0/scratch/shared:/shared gpt-neox bash
        cd /shared
        git clone https://github.com/EleutherAI/gpt-neox.git gpt-neox-main
        cd gpt-neox-main
        pip install -r requirements/requirements.txt
        pip install -r requirements/requirements-wandb.txt
        pip install -r requirements/requirements-flashattention.txt
        ```
    3. Install APEX
        ```bash
        cd /shared
        git clone https://github.com/NVIDIA/apex
        cd apex
        # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
        pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
        ```
    4. Install Fusion kernels
        - Had to install older setuptools due to breaking change
            ```bash
            pip install setuptools==69.5.1
            ```
        - Continue kernel install
            ```python
            from megatron.fused_kernels import load
            load()
            ```