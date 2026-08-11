### 🌀 ⏱️ **kEDMD-DDE: Koopman Representations of Nonlinear Delay Differential Equations**

 🚀 **A data-driven Koopman surrogate for nonlinear DDEs, with deterministic, interpretable error bounds instead of heuristic delay embeddings.**

 ⚡ GPU-accelerated kEDMD • ✅ Provable error decomposition • 🔧 Ships with data

[![Paper](https://img.shields.io/badge/Accepted-IEEE%20CDC%202026-darkred)](https://arxiv.org/abs/2604.03086)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)]()
[![Stars](https://img.shields.io/github/stars/santoshrajkumar/koopman-dde-kEDMD?style=social)]()

<h4><code>kedmd_dde</code> — A <b>self-contained PyTorch package</b> for <b>kernel extended dynamic mode decomposition (kEDMD)</b> on <b>delay differential equations</b>. Built on <b>history discretization + reconstruction</b>, so an infinite-dimensional delay system becomes a <b>finite-dimensional linear surrogate</b> with <b>explicit error guarantees</b>.</h4>

<p align="center">
  <a href="https://github.com/santoshrajkumar/koopman-dde-kEDMD">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo_for_updates!-red?style=for-the-badge">
  </a>
</p>

<p align="center">
  ✅ Deterministic bounds: discretization + interpolation + regression, separately
  ✅ Runs on GPU or CPU with no change to the calling code
  ✅ Datasets ship inside the wheel — every figure reproduces from a bare checkout
</p>


## 🌟 Key Features

✅ **Finite-dimensional Koopman realization for DDEs**
✅ **Deterministic error guarantees**
✅ **kEDMD with Wendland RBF kernels**
✅ **State reconstruction from lifted coordinates**
✅ **GPU-first, CPU-fine**
✅ **Reproducible by construction**

## ❗ Why It Matters

Delay differential equations evolve on an **infinite-dimensional** phase space of history segments. Koopman methods for delay systems have therefore leaned on **heuristic delay embeddings** that compress that history into finitely many samples, **without quantifying the error they induce**. That is exactly the guarantee prediction and control need.

**This work closes the gap** by making the compression explicit : a sampling operator, a reconstruction operator, and a kEDMD surrogate on the resulting finite-dimensional domain. This means:
✅ **A linear surrogate on a Banach space of history functions**
✅ **An error bound you can read term by term, and shrink term by term**
✅ **Convergence in both discretization resolution `M` and data density `p`**
✅ **Reconstruction of the discretized state, with its own bound.**

## 🧠 Paper

This work is based on:

> **Rajkumar, S.M.**, Barman, D., Singh, K.V. and Goswami, D., 2026. On Data-Driven Koopman Representations of Nonlinear Delay Differential Equations. *2026 65th IEEE Conference on Decision and Control (CDC)*, accepted. [[ArXiv]](https://arxiv.org/abs/2604.03086)

If you use this repository, **please cite us** 🙏

```bibtex
@article{rajkumar2026data,
  title={On Data-Driven Koopman Representations of Nonlinear Delay Differential Equations},
  author={Rajkumar, Santosh Mohan and Barman, Dibyasri and Singh, Kumar Vikram and Goswami, Debdipta},
  journal={arXiv preprint arXiv:2604.03086},
  year={2026}
}
```

## 🔧 Installation

*Virtual environment recommended

```bash
git clone https://github.com/santoshrajkumar/koopman-dde-kEDMD.git
cd koopman-dde-kEDMD
python3 -m venv dde_env
source dde_env/bin/activate
pip install -r requirements.txt
```

Installing the package is optional — every script prepends the repository root to `sys.path`, so they run from a bare checkout. If you do install it, the scripts still pick up the copy here:

```bash
pip install -e .
```

The default `torch` wheel bundles CUDA. For a **CPU-only** install (much smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## ⚡ Quick Demo

```bash
python examples/learn_and_predict.py
```

```python
from kedmd_dde import load_dataset, run_experiment, summarize_errors

ds = load_dataset("raw_scalar.npz", M=3)                     # embed, split, pair
res = run_experiment(ds, {"n_centers": 100, "rho": 0.6,      # fit + roll out the test set
                          "epsilon": 1.0, "d_neighbors": 25})
summary, = summarize_errors([res])
```

Each run opens by naming the device it is about to use:

```
[kedmd_dde] device: cuda:0
```

That is `print_device_info(brief=True)`. Dropping `brief` gives the full report — the card, its compute capability and free memory, the torch/CUDA/cuDNN versions, and the dtype with a note when FP64 is throttled on that architecture.

### ⚠️ **Important Dependency Notice**

* Fitting, predicting and every shipped figure need **only** `torch`, `numpy`, `scipy` and `matplotlib`.
* **`jitcdde` is not required.** Both trajectory records ship inside the package; it is needed *only* to integrate new ones.
* `jitcdde` compiles the right-hand side to C at run time, so a **C compiler (gcc/clang) on `PATH`** is recommended; without one it falls back to a slower pure-Python evaluation.
* **Quick Setup Checklist**
- Install `requirements.txt` ✅
- (Optional) `pip install -r requirements-gen.txt` for regeneration ✅
- (Optional) `KEDMD_DEVICE` / `KEDMD_DTYPE` to pin device and precision ✅
- Headless plotting: `export MPLBACKEND=Agg` ✅

## 📈 Reproducing the Paper Figures

Every figure in the paper comes from one script:

```bash
python examples/pub_gen.py
```

| function it calls | what it sweeps | output |
| --- | --- | --- |
| `m_vary` | history resolution `M`, centers matched on fill distance | `scalar_M_vary.pdf` |
| `p_vary` | kernel-center count `p` at fixed `M` | `p_vary.pdf` |
| `rho_vary` | local-fit radius `ρ`, radius-only neighbourhoods | `rho_vary.pdf` |
| `scalar_phase` | one fit — true vs predicted `x(t)`, plus RMSE | `scalar_phase_new.pdf` |
| `planar_phase` | one fit — true vs predicted `x₁(t)`, `x₂(t)`, plus RMSE | `planar_phase_new_x{1,2}.pdf` |

Figures land in `figures/`. Each of those is an argument-free function in `kedmd_dde/Utils/pub_gen.py` with its settings as module constants, so a variation is an edit at the top of the block rather than a flag. `examples/pub_gen.ipynb` is the same run as a notebook.

The script closes with `gen_data()`, which asks before writing anything — answer no and the shipped records are left exactly as they are.

## 📊 Convergence of the Learned Predictor

Step-wise mean prediction error `μ_{z_k}` over the 100 held-out test trajectories, scalar Hill-type DDE (`τ_d = 1 s`); the shaded band spans worst to best.

<table align="center">
  <tr>
    <td align="center">
      <img src="media/scalar_M_vary.png" width="300px" style="border:1px solid #d0d7de;border-radius:6px;"/><br>
      <sub>Increasing <b>M</b>, fill distance matched</sub>
    </td>
    <td align="center">
      <img src="media/p_vary.png" width="300px" style="border:1px solid #d0d7de;border-radius:6px;"/><br>
      <sub>Increasing centers <b>p</b></sub>
    </td>
    <td align="center">
      <img src="media/rho_vary.png" width="300px" style="border:1px solid #d0d7de;border-radius:6px;"/><br>
      <sub>Decreasing local radius <b>ρ</b></sub>
    </td>
  </tr>
</table>

Current value of the state function, recovered from the lifted coordinates:

<table align="center">
  <tr>
    <td align="center">
      <img src="media/scalar_phase_new.png" width="300px" style="border:1px solid #d0d7de;border-radius:6px;"/><br>
      <sub>Scalar DDE, <code>x(t)</code></sub>
    </td>
    <td align="center">
      <img src="media/planar_phase_new_x1.png" width="300px" style="border:1px solid #d0d7de;border-radius:6px;"/><br>
      <sub>Tumour–immune model, <code>x₁(t)</code></sub>
    </td>
    <td align="center">
      <img src="media/planar_phase_new_x2.png" width="300px" style="border:1px solid #d0d7de;border-radius:6px;"/><br>
      <sub>Tumour–immune model, <code>x₂(t)</code></sub>
    </td>
  </tr>
</table>

### 🔍 Highlights

- ✅ **Error falls monotonically with `M`** at matched fill distance — the discretization term `R_se·δ(M)` shrinking as predicted
- ✅ **Error falls with `p`** — denser centers, smaller fill distance `h_X`
- ✅ **Error falls with `ρ`** — the regression term is quadratic in the local radius
- ✅ **Both systems reconstruct** the current state from lifted coordinates to plotting accuracy

## ⤴️ The Method

For snapshot pairs `(z_k, z_{k+1})` of the delay-embedded state:

1. Place `p` kernel centers `C`, farthest-point sampled from the training cloud.
2. At each center `c`, fit `z⁺ ≈ a + B (z - c)` to the snapshots inside `B(c, ρ)`. The intercept `a` estimates the flow map at the center, `F̂(c)`.
3. Build the Koopman matrix `A = K_FX^T (K_X + reg·I)⁺` from the Wendland kernel matrices `K_X = k(C, C)` and `K_FX = k(F̂(C), C)`.

Prediction lifts `z₀` to `ψ = k(C, z₀)`, iterates `ψ ← A ψ`, and decodes with `Wᵀ = Cᵀ (K_X + reg·I)⁺`.

### Kernel degree

The kernel is the C² Wendland function `φ(r) = (1 - r/ε)₊ⁿ (n r/ε + 1)` with `n = wendland_degree(nM) - 1 = ⌊nM/2⌋ + 3`. The degree is **not fixed**: a Wendland function built for dimension `d` is positive definite on `R^d` and every smaller space, but not in general on a larger one — and strict positive definiteness is what makes `K_X` invertible. `wendland_matrix` takes the dimension from the points it is given, so a model cannot be fitted at one degree and evaluated at another.

The scalar runs (`nM = 2, 3`) land on `n = 4`, the familiar `(1-r)⁴(4r+1)`. The planar runs lift to `nM = 4` and `6` and get `n = 5` and `6`, which are narrower at the same `ε` — so **`ε` is worth re-tuning as `nM` grows**, together with the center count that keeps the balls populated.

### Checking Assumption 3

Assumption 3 requires every local data matrix `Z̄_ℓ ∈ R^{(nM+1)×d}` to have full row rank `nM+1`, with the `d` neighbours drawn from inside `B(z_ℓ, ρ)`. `check_assumption3` scores exactly the neighbourhoods `build_kedmd` would form:

```python
from kedmd_dde import check_assumption3, farthest_point_sampling, load_dataset

ds = load_dataset("raw_planar.npz", M=2)
C = farthest_point_sampling(ds.X, n_centers=1200, seed=0)
print(check_assumption3(ds.X, C, rho=0.1, d_neighbors=25))
```

or inline while fitting, which puts the verdict in `model.info["assumption3"]`:

```python
model = build_kedmd(ds.X, ds.Y, n_centers=1200, rho=0.5, assumption3=True)
```

It separates the two ways the assumption fails:

- **the ball is too thin** — fewer than `nM+1` points inside `B(z_ℓ, ρ)`, so `Z̄_ℓ` cannot reach full rank at all. `build_kedmd` then steps outside the ball, which restores the rank but breaks `max_j ‖δ_{ℓ,j}‖ ≤ ρ`, the inequality the `R_d ρ²` term rests on. `rho_eff` is the radius the bound actually holds with;
- **the ball is degenerate** — enough points, but on a lower-dimensional set through `z_ℓ`.

Full rank is necessary but not sufficient for a *useful* bound, so the report also gives `σ_min(Z̄_ℓ)` and `Z̄ = max_ℓ ‖Z̄_ℓ†‖`. A neighbourhood that is nominally full rank with `σ_min` near round-off satisfies the assumption while making the bound vacuous.

### Matching the fill distance across `M`

What the error bounds are stated in is the fill distance `h_X`, not the number of centers. The lifted space gains a dimension with every delay coordinate, so a fixed `p` covers it more thinly as `M` rises. `matched_center_configs`, which `m_vary` uses, pins `h_X` instead and gives each `M` the fewest centers that reach it.

### Neighbourhood selection

- **`d_neighbors`** — cap on how many points inside the ball are used. `None` uses every point in `B(c, ρ)`, so `ρ` alone sets the locality. This matters: on these records the 25 nearest neighbours sit far inside even a small ball, so with the cap on, `ρ` is inert over a wide range. `rho_vary` therefore defaults to radius-only.
- **`rank_recovery`** — what to do when the local design matrix is rank deficient. `"multiplier"` jumps to `max(3k, nM+3)` nearest points; `"expand"` grows the neighbour set one point at a time.

## 💻 Device and Precision

| Variable | Values | Default |
| --- | --- | --- |
| `KEDMD_DEVICE` | `cuda`, `cuda:1`, `cpu` | CUDA if available, else CPU |
| `KEDMD_DTYPE` | `float64`, `float32` | `float64` |

```bash
KEDMD_DEVICE=cpu python examples/learn_and_predict.py
KEDMD_DTYPE=float32 python -m kedmd_dde.Utils.pub_gen planar_phase
```

`float64` is the default because the estimator pseudo-inverts a kernel matrix that is close to singular. On consumer NVIDIA cards FP64 throughput is a fraction of FP32 (1/64 on Ampere GeForce), so `float32` is substantially faster where the precision loss is acceptable.

Tensors stay on the device from loading through to the error summaries; NumPy appears only at file I/O, in data generation, and after `to_numpy` for matplotlib.

## 🗂️ Layout and Data

```
kedmd_dde/      KEDMD_DDE.py -- kernels, center placement, the estimator
                Utils/ -- device policy, data loading, generation, sweeps, plotting, pub_gen
                data/ -- raw_scalar.npz, raw_planar.npz: trajectories, not embeddings
examples/       learn_and_predict.py, pub_gen.py, pub_gen.ipynb
data/           records you generate yourself, searched after the packaged ones
figures/        created on first run; every script writes its output here
media/          the figures above, as PNG
```

The delay embedding, the train/test split and the successor pairing all happen at **load time**, so the records store each solution once instead of repeating every sample `2M` times. That is what keeps them at 5 MB — small enough to ship inside the package, so `load_dataset("raw_scalar.npz", M=3)` works from an installed wheel and not only from a checkout.

### Regenerating the data

```bash
pip install -r requirements-gen.txt      # adds jitcdde; needs a C compiler on PATH
python -m kedmd_dde.Utils.pub_gen --data
```

```python
from kedmd_dde.Utils.pub_gen import gen_data
from kedmd_dde import load_dataset

paths = gen_data()                      # or gen_data("scalar", n_traj=4000, dt=0.05)
ds = load_dataset(paths["scalar"], M=3)
```

The step `dt` decides which `M` a record can represent — `M - 1` must divide `τ/dt`. The shipped records give `M ∈ {2, 3, 6, 11}` for the scalar system and `{2, 3, 5, 6, 11}` for the planar one.

A record missing from `kedmd_dde/data/` is generated into it. A record already there is **never overwritten**: the prompt offers to write a fresh one into `data/` beside the package instead. Load that copy by the returned path — `data_path` searches the package first, so the bare name still opens the shipped record.

## 🧪 The Systems

| system | equation | `τ_d` |
| --- | --- | --- |
| scalar, Hill-type [Glass et al., 2021] | `ẋ = 1/(1 + x(t-τ)²) - x` | 1.0 s |
| planar, tumour–immune [Rihan et al., 2014] | non-dimensionalised two-state delayed model | 1.636 s |

---

🚀 **Coming next**
🔧 **Closed-loop control synthesis on the lifted surrogate**
🔧 **Extensions to high-dimensional networked systems with delays**
🔧 **Multi-step bound reporting alongside the rollouts**

This work is supported by **NSF-DMS Math-DT under Grant 2529302**.

If you find this project useful, please ⭐ star the repo and follow — your support drives development!

<p align="center">
  <a href="https://twitter.com/intent/tweet?text=Koopman%20operator%20learning%20for%20delay%20differential%20equations%20with%20deterministic%20error%20bounds%20%E2%9A%A1%20kEDMD%20on%20GPU%2C%20open%20source%20by%20%40SantoshRajkumar&url=https://github.com/santoshrajkumar/koopman-dde-kEDMD">
    <img src="https://img.shields.io/badge/Share_on_Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"/>
  </a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/santoshrajkumar/koopman-dde-kEDMD">
    <img src="https://img.shields.io/badge/Share_on_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
</p>

---
