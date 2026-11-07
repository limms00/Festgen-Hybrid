# EDM score 계속학습과 0.8/0.2 혼합 교사 비교

## 결과

두 300-update 후보를 개발 평가 후 고정하고 같은 로컬 TA 절차로 각각 평가했다. 이는 서버 채점이 아니다. 3020 validation reference, 3020 generated, 20/class, batch32, seed42+batch index, CUDA bf16, clean-fid, serial DataLoader, `evaluate.py` SHA256 f201320f3505ad478096313eedf657388a0c6aca4af456095509e13807228adb.

| 4-NFE frozen model | local TA FID | dev seed42 | dev seed1042 | dev seed2042 |
|---|---:|---:|---:|---:|
| EDM score initial champion | 29.5316 | 46.313422375 | 46.202267499 | 46.402249619 |
| Pure EDM continuation, 300 | 29.0699 | 46.245248674 | 46.154483928 | 46.335541898 |
| 0.8 EDM + 0.2 VP hybrid, 300 | **28.8481** | 46.434376897 | 46.327677850 | 46.521434051 |

Hybrid TA is 0.6835 below initial, 0.2218 below pure, and below 29. Pure TA is 0.4617 below initial. However hybrid dev FID is **higher** than pure by 0.189128223 / 0.173193922 / 0.185892153. Thus metrics disagree: the single validation run favors hybrid while all three shared-reference dev seeds favor pure. No general claim that mixing adds robust value is supported. The validation set was used repeatedly to compare candidates, and the EDM teacher had validation-based historical checkpoint/sampler selection. The EDM teacher also trained on all 755 train-derived dev reference images. Thus neither repeated validation nor dev results are untouched independent generalization evidence. All models share the initial checkpoint and the new arms use a single training seed 680101, reused from the parent experiment; no independent-training-seed evidence.

## Controls and math

Both branches load `pilot_edm_200_final.ckpt` SHA256 b203f0ee6f54a7097865e57e8db3af92f92a60884a21a1f4b2eaaf1dfa85e0e8. Original fake-score teacher reset and two fake updates per generator after warmup100, fresh optimizers, 300 generator updates, EMA .99, batch16/micro4, generator/fake lr1e-6/1e-4, anchor weight1 to **original full DMD** checkpoint 053195fe1da3c4f884cb34feaaeb878030744403768f3a7fc7a4fe5ea3c99d5a, CFG2, uniform t50..596, train-image-frequency class labels. This is a fresh-optimizer restart, not full optimizer-state continuation. The separate branch choice is pure EDM x0 vs 0.8 EDM x0 + 0.2 VP x0; both teacher predictions computed for both arms at identical x,t,y. Fixed 0.8/0.2 was specified before validation, not tuned from validation.

For VP x=alpha*x0+sigma*epsilon and velocity v, x0=alpha*x−sigma*v and epsilon=sigma*x+alpha*v. EDM coordinates y=x/alpha, q=sigma/alpha, scoreVP=(D−y)/(alpha*q²), epsilon=(y−D)/q. For fixed x,t, interpolation in x0 is exactly the corresponding epsilon and score interpolation. FP64 algebra tests passed; both 2-step GPU smokes passed with nonzero gradients through each of four calls, same full input fingerprints, and GPU conversion identity assertions. For all 300 updates, cumulative SHA256 of labels, entire initial noise, timesteps and diffusion noise matched, final 605288de4f18fb9c2f5176798fa3334e2f77eac2d6d02840550332cea0489bca. Other GPU nondeterminism is not ruled out.

## Stability and preservation

Both runs completed 300/300. Allocated peak VRAM pure/hybrid 4,753,659,392 / 4,753,856,000 bytes, training including warmup 1173.69 / 1113.76 seconds. Mean teacher x0 disagreement RMS over training .17915/.17907; final 25-step fake loss .018185/.017993 and generator gradient norm 1.1704/1.1798. On eight fixed different-class examples, pure/hybrid output standard deviations .72834/.72812, paired RMS .90178/.90098. These are small proxies, not evidence for absence of mode collapse. The two fixed sample strips look nearly identical; `pure_300/fixed_samples.png` and `hybrid_300/fixed_samples.png`.

Strict original Model loader and four backbone calls were verified after saving; each has 48,883,107 parameters. Both candidates are in volatile `/run`, no independent permanent copies were made due 2.25GiB root floor:

* `/run/user/0/fastgen_edm_score_hybrid_20261105/pure_300_final.ckpt` SHA256 281b97c148e6789ef37c38137d783abb9b8f83e65b782545e2af9be78efd3f08.
* `/run/user/0/fastgen_edm_score_hybrid_20261105/hybrid_300_final.ckpt` SHA256 36a457612d241a0a462c37d2a73a73f29257fb4d297dbd296851a2b0c1b66.

Existing 29.5316 checkpoint remains `/run/user/0/fastgen_edm_score_transfer_20261105/pilot_edm_200_final.ckpt` SHA256 b203f0ee6f54a7097865e57e8db3af92f92a60884a21a1f4b2eaaf1dfa85e0e8 and is also volatile. All three can vanish after reboot; transfer directly to a permitted persistent destination without writing extra copies to root, then verify hashes. Root free 2,499,293,184 and /dev/shm free 2,525,720,576 bytes after run, >2.25GiB each. No existing file deleted or overwritten.

The parent DMD used uniform synthetic class sampling, and upstream checkpoint training history is incomplete. The EDM teacher was selected with historical validation use. These raise unresolved course compliance questions. Neither validation statistics nor resulting FID were used to alter training sampling, teacher mixing or objectives. TA clarification remains required.
