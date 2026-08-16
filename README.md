# ANGKOR ⚛️ 🇰🇭

### Advanced Neutron Group-diffusion K-eigenvalue Of Reactors Analysis

*Named after Angkor Wat — the greatest achievement of Khmer civilization.*

An open-source deterministic reactor physics code developed at the **Institute of Technology of Cambodia**. It solves the multigroup neutron diffusion equation on 2D rectangular geometry, with CMFD acceleration, and is written entirely in Python to be read.

If you want to understand what a diffusion solver actually does, line by line, this is a reasonable place to start.

**Status** — `v0.6.0` · multigroup diffusion with verified CMFD acceleration · MIT licensed

---

## Verification

| Check | ANGKOR | Reference | Difference |
|---|---|---|---|
| Infinite medium k∞ (analytic) | 1.33026789 | 1.330268 | 0.011 pcm |
| 1D bare slab (analytic) | 1.305447 | 1.305446 | 0 pcm |
| PWR 2D slab, CMFD vs unaccelerated | 1.17978975 | 1.17978975 | 0 pcm |
| CMFD exactness (coarse reproduces fine) | — | — | 0.00000 pcm |
| IAEA 2D PWR quarter-core | 0.99837 | 1.02959 | −3086 pcm |

**Convergence:** 130 → 14 outer iterations with CMFD enabled, no under-relaxation.

**On the IAEA result.** The −3086 pcm discrepancy is expected. Roughly 2000 pcm comes from the diffusion approximation itself (P1 versus transport), and roughly 1000 pcm from a 2-group energy structure too coarse to capture spectral effects near the fuel–reflector interface. Closing the gap requires transport-corrected diffusion coefficients or more groups in the reflector — both on the roadmap.

```bash
python main.py input/iaea_2d.yaml
```

---

## Capabilities

- Multigroup neutron diffusion — 1G, 2G, 4G, and arbitrary G
- 2D rectangular geometry, 5-point finite difference stencil
- Harmonic face coupling — exact current continuity at material interfaces
- CMFD acceleration — consistent, no damping
- Vacuum and reflective boundary conditions
- Power iteration with direct sparse solves (SciPy `spsolve`)
- YAML input: geometry, materials, and solver settings in one file
- Geometry visualiser, flux maps, power distribution, centerline profiles
- pytest suite including analytic and self-consistency checks

---

## Quick start

```bash
uv venv
uv pip install -r requirements.txt
python main.py input/iaea_2d.yaml
pytest -v
```

```
k-eff = 0.998366
Results saved to: output/iaea_2d/
```

---

## Method notes

**Face coupling.** Interior faces use the harmonic mean `2·Da·Db/((Da+Db)·h²)`, which follows from treating the two half-cells as diffusion resistances in series. An arithmetic or own-cell mean makes the matrix non-symmetric at material interfaces and fails to conserve current across them.

**CMFD.** Nonlinear diffusion acceleration. The correction factor `d̂` closes the coarse face current against the fine-mesh current, so the coarse operator reproduces the fine solution exactly when handed it. This is enforced by a unit test. No under-relaxation is used or needed.

**Boundary coupling.** Coarse vacuum faces use the flux-weighted form `E_b = Σ(D·φ) / (H²·φ̄)`, summed over the fine faces spanning the coarse face. A volume-averaged coefficient under-leaks by roughly the refinement factor.

**Coarse mesh limit.** The coarse cell must not exceed roughly two diffusion lengths of the most tightly coupled group, `L = √(D/Σr)`. For the PWR slab benchmark L₂ ≈ 2.1 cm, so `rf=4` is stable while `rf=20` diverges. Defaults are `rf=4`, `cmfd_interval=2`.

---

## Example input

```yaml
title: "IAEA 2D PWR Benchmark"

geometry:
  type: 2D_rectangular
  domain_x: 170.0
  domain_y: 170.0
  nx: 170
  ny: 170

regions:
  - {name: fuel_core, x_min: 0,  x_max: 80,  y_min: 0, y_max: 80,  material: fuel1}
  - {name: reflector, x_min: 80, x_max: 170, y_min: 0, y_max: 170, material: reflector}

materials:
  fuel1:
    D1: 1.500
    D2: 0.400
    sigma_a1: 0.0100
    sigma_a2: 0.0850
    sigma_s12: 0.0200
    nu_sigma_f1: 0.000
    nu_sigma_f2: 0.135

boundary_conditions:
  left:   reflective
  bottom: reflective
  right:  vacuum
  top:    vacuum

solver:
  max_iterations: 1000
  convergence: 1.0e-6
```

---

## Project structure

```
ANGKOR/
├── angkor/
│   ├── geometry_2d.py     # 2D rectangular geometry engine
│   ├── input_reader.py    # YAML input reader
│   ├── solver_2d.py       # 2-group diffusion solver
│   ├── solver_mg.py       # G-group multigroup solver
│   ├── cmfd.py            # CMFD acceleration
│   ├── output_2d.py       # Flux maps and power distribution
│   ├── geometry.py        # 1D geometry (legacy)
│   └── solver_1d.py       # 1D diffusion solver (legacy)
├── input/                 # Input files
├── output/                # Simulation results
├── tests/                 # Unit and verification tests
├── benchmarks/            # Benchmark cases
├── docs/                  # Theory manual
├── main.py                # Entry point
└── requirements.txt
```

---

## Known limitations

- Coarse cross sections are volume-averaged rather than flux-weighted. Exact only when coarse cells are materially homogeneous.
- CMFD instability at large refinement factors is detected but not raised; the solver can return a converged-looking wrong answer.
- No transport correction, no resonance self-shielding, no nodal homogenisation.

---

## Roadmap

Organised around three questions: **can it converge faster, can it model more physics, can it be validated against something hard?**

### Phase 1 — Core solver

- [x] 1D slab finite difference solver
- [x] 2D rectangular geometry, 5-point stencil
- [x] 2-group and G-group multigroup diffusion
- [x] IAEA 2D PWR quarter-core benchmark
- [x] Coarse Mesh Finite Difference (CMFD) acceleration
- [ ] Flux-weighted coarse homogenization
- [ ] Chebyshev or JFNK acceleration for eigenvalue convergence
- [ ] Cylindrical 1D geometry (annular pins, TRIGA-type problems)
- [ ] 3D Cartesian extension

### Phase 2 — Neutronics accuracy

- [ ] Transport correction for diffusion coefficients (B1 or P1 leakage)
- [ ] Adjoint flux solver — perturbation theory, reactivity worth, sensitivity and uncertainty analysis (SCALE TSUNAMI-equivalent workflow)
- [ ] Reflector discontinuity factors (Koebke equivalent homogenization)
- [ ] SP3 approximation — better accuracy near material interfaces without full transport
- [ ] Multigroup cross section library reader (WIMS/SCALE few-group formats)
- [ ] Benchmark suite: ANL, OECD/NEA C5G7, KAIST 3D PWR

### Phase 3 — Core simulation

- [ ] Nodal Expansion or Analytic Nodal Method — orders of magnitude faster than fine-mesh FD on full-core LWR problems, and what SIMULATE and DIF3D actually use
- [ ] Burnup and isotopic depletion (Bateman, matrix exponential)
- [ ] Reactivity feedback: Doppler, moderator temperature and density, boron
- [ ] Control rod cusping correction
- [ ] Shutdown margin and ejected rod worth

### Phase 4 — Research features

- [ ] ENDF/B processing pipeline, or an NJOY output interface
- [ ] Simplified thermal hydraulics coupling (single channel, COBRA-like)
- [ ] Stochastic uncertainty quantification (SCALE/Sampler-compatible)
- [ ] ML surrogate for cross section parameterization versus burnup, temperature, and boron
- [ ] Verification against OpenMC and Serpent on identical definitions

### Out of scope by design

Full Monte Carlo transport, resolved resonance processing, and production-grade thermal hydraulics. Those problems are solved by OpenMC, Serpent, and RELAP/TRACE. ANGKOR's role is to be a readable, modifiable diffusion solver that a graduate student can actually debug.

---

## Why open source

Commercial reactor physics codes such as CASMO, SIMULATE, DIF3D, PARCS are accurate but not free. SCALE is excellent but requires a license and considerable setup. OpenMC is open source and use Monte Carlo method.

If you find a bug, open an issue. If you use it for a class or a paper, a citation is appreciated.

---

## Collaboration

If you work on reactor physics, neutronics methods, or nuclear data and particularly on nodal methods, transport correction, or cross section library integration let get in touch.

The ITC group is also developing ML surrogate models for SMR k-effective prediction using CASMO/SIMULATE-generated training data. If that overlaps with your work, reach out directly. ITC group currently has only me :( 

---

## Author

**MUTH Boravy**
PhD, Nuclear Engineering
Institute of Technology of Cambodia, Phnom Penh, Cambodia 🇰🇭
[github.com/muthboravy007](https://github.com/muthboravy007)

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).

## Citation

```
MUTH Boravy, "ANGKOR: Advanced Neutron Group-diffusion K-eigenvalue Of
Reactors Analysis", Institute of Technology of Cambodia, 2025.
https://github.com/muthboravy007/ANGKOR
```
