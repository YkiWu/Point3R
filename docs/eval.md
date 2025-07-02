# Evaluation

## Data Preparation
Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare the evaluation datasets.

## Scripts

Our evaluation code follows [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/eval.md).

### 3D Reconstruction

```bash
bash eval/mv_recon/run.sh
```

Results will be saved in `eval_results/mv_recon/${model_name}_${ckpt_name}/logs_all.txt`.

### Monodepth

```bash
bash eval/monodepth/run.sh
```
Results will be saved in `eval_results/monodepth/${data}_${model_name}/metric.json`.

### Video Depth

```bash
bash eval/video_depth/run.sh 
```
Results will be saved in `eval_results/video_depth/${data}_${model_name}/result_scale.json`.

### Camera Pose Estimation

```bash
bash eval/relpose/run.sh 
```
Results will be saved in `eval_results/relpose/${data}_${model_name}/_error_log.txt`.


