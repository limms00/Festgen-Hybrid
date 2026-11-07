# Festgen-Hybrid — 4-NFE VP score-transfer research handoff

**Research handoff; course compliance is unresolved.** This class-conditional 64×64 Pokémon generator uses the exact checkpoint locally evaluated at **FID 28.8481** under the TA-provided procedure. This is a locally measured and reproduced score, not an official TA-server score.

## Lineage and method

Midterm VP Few-NFE checkpoint → 1,000-update full-trajectory DMD → 200-update EDM real-score transfer → 300-update hybrid real-score continuation. The 4-step student is a 48,883,107-parameter VP velocity U-Net. All four generator transitions receive gradients during DMD training. In the final training stage the teacher target is 0.8 EDM and 0.2 VP **clean-image prediction** at the same noisy state, class and timestep; it is not a parameter or output-image ensemble. An original-DMD output MSE anchor is used during training.

For x_t = αx₀ + σε, the EDM denoiser is evaluated at y=x_t/α and q=σ/α; its score transforms to VP coordinates as (D(y,q)−y)/(αq²). VP and EDM teachers, fake-score networks and anchor are training-only: deployed inference uses exactly four backbone calls and no training checkpoint or data.

The 200- and 300-update stages use seed 680101, batch 16 / microbatch 4, generator/fake AdamW learning rates 1e−6 / 1e−4, 100 fake warmup updates, two fake updates per generator step, EMA 0.99, CFG 2 during training and timestep interval 50–596. Both use new optimizers; this is not a full optimizer-state resume. See `training/` for historical source, config and loss logs. Those files depend on the original research environment, datasets and teacher weights, which are not bundled; independent training reproducibility is **not** claimed. EDM progressive distillation was a separate experiment and is not an ancestor of this checkpoint.

## Results

| Model | Local TA-procedure FID | Dev FID, seed 42 / 1042 / 2042 |
|---|---:|---|
| Original Full-trajectory DMD | 30.3432 | 46.491548 / 46.427268 / 46.651664 |
| EDM score transfer, 200 | 29.5316 | 46.313422 / 46.202267 / 46.402250 |
| Hybrid, 300 — **this checkpoint** | **28.8481** | 46.434377 / 46.327678 / 46.521434 |

The local TA procedure used 3020 generated / 3020 validation images, 20 per class, batch 32, seed 42 + batch index, CUDA BF16 generation and FP32 clean-fid. A serial DataLoader wrapper was used with unchanged `evaluate.py`; the four-decimal FID was independently rerun and matched. The train-derived dev reference had 755 images, was exposed to the EDM teacher, and is not an independent test. Pure EDM performed better than Hybrid on those three dev seeds. Validation has been examined repeatedly; selection bias and single-training-seed limitations remain.

![Fixed-seed sample grid](results/hybrid_samples.png)

## Install, verify and sample

Requires Python 3.10 and a compatible PyTorch build; the tested environment used PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124 on an NVIDIA A100. Pull the actual LFS object after cloning:

```bash
git lfs install
git clone https://github.com/limms00/Festgen-Hybrid.git
cd Festgen-Hybrid
git lfs pull
python -m pip install -r requirements.txt
python verify.py --device cuda
python sample.py --device cuda --labels 0 --seed 42 --output samples.png
```

The URL above is the **intended** destination and is not a claim that the repository has been created. `--device cpu` is supported but slow for batch 32. Labels are course class IDs 0–150; the mapping is `configs/category_to_id.json`. The verifier checks strict loading, SHA256, 48,883,107 parameters, exactly four calls, batch32 shape/dtype/range and deterministic seeded outputs. Use `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux. Only load trusted PyTorch checkpoints.

The checkpoint is `checkpoints/few_nfe.ckpt`, SHA256 `36a457612d241a0a462c37d2a73a33a73f29257fb4d297dbd296851a2b0c1b66`. The compatible `ModelFewNFE` interface is in `model.py`; integration into the official project requires retaining the TA-provided base class and evaluator and checking the final archive in the actual course environment. `teacher/` contains architecture/schedule definitions, not an inference-time teacher model or weights. No dataset is needed for inference.

## Compliance and provenance

The earlier Full-trajectory DMD stage sampled synthetic class labels uniformly, unlike empirical training-image frequency. The original checkpoint's complete training history is not verified. The EDM teacher's historical sampler/CFG selection used validation, and multiple candidate models were compared on the same validation reference. The fixed 0.8/0.2 weight was specified before this candidate's evaluation; validation statistics must not be used as a retrospective justification for target-distribution choices. **TA clarification is required before final submission.** Do not claim these research weights are approved or an untouched independent test.

Research builds on the team's original VP Few-NFE model, Distribution Matching Distillation (Yin et al., 2024), and EDM (Karras et al., 2022). This is a FastGen-specific adaptation, not a complete reproduction of those papers.
