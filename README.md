# ProgCoPL
The ProgCoPL model was trained using a single NVIDIA 3090 GPU with a batch size of 4 for each dataset. The PyTorch version is 3.8, and each training run consists of 5 epochs.
All result is stored in `/output`

# Training and Evaluation
#### (1) Base-to-Novel
The default training settings are provided in config file at `configs/trainers/ProgCoPL/vit_b16_c2_ep5_batch4_2ctx.yaml`.
```bash
#seed=1
bash scripts/progcopl/base2new_train_progcopl.sh imagenet 1
bash scripts/progcopl/base2new_test_progcopl.sh imagenet 1
```


#### (2) Cross-Dataset Transfer
We provide cross-dataset config for ProgCoPL: `configs/ProgCoPL/vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml`.
```bash
# seed=1 
bash scripts/progcopl/xd_train_progcopl.sh imagenet 1

bash scripts/progcopl/xd_test_progcopl.sh caltech101 ${SEED}

```

#### (3) Domain Generalization 
```bash
    bash scripts/progcopl/xd_test_progcopl.sh imagenetv2 ${SEED}
    bash scripts/progcopl/xd_test_progcopl.sh imagenet_sketch ${SEED}
    bash scripts/progcopl/xd_test_progcopl.sh imagenet_a ${SEED}
    bash scripts/progcopl/xd_test_progcopl.sh imagenet_r ${SEED}
```