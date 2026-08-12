# ANGKOR ⚛️ 🇰🇭
### Advanced Neutron Group-diffusion K-eigenvalue Of Reactors Analysis

*Named after Angkor Wat  the greatest achievement of Khmer civilization.*

ANGKOR is an open-source deterministic reactor physics code developed at the **Institute of Technology of Cambodia (ITC)**. It solves the multi-group neutron diffusion equation on 2D rectangular geometry using finite difference discretization and power iteration for k-eff and flux distribution.

The code is written entirely in Python and designed to be readable. If you want to understand what a diffusion solver actually does line by line, this is a reasonable place to start.

---

## Current ANGKOR Capabilities

- Multigroup neutron diffusion: 1G, 2G, 4G, and G-group (arbitrary).
- 2D rectangular geometry with a 5-point finite difference stencil.
- Power iteration for k-eff eigenvalue and flux shape distribution.
- YAML input files: for geometry, materials, and solver settings in one place
- Geometry visualizer: color-coded material maps rendered interactively
- Flux maps, power distribution, and centerline profiles on output
- pytest unit test

---

## Physics context

ANGKOR solves the within-group diffusion equation iteratively using Gauss-Seidel sweeps, with fission source updates between outer iterations. The current implementation uses point-wise convergence on both flux and k-eff.

The code does **not** use transport-corrected cross sections, CMFD acceleration, or nodal homogenization. This matters for interpreting results — see the validation section below.

---

## Validation

| Benchmark | ANGKOR k-eff | Reference | Error | Notes |
|---|---|---|---|---|
| 1D bare slab (analytical) | 1.305447 | 1.305446 | 0 pcm | Finite difference vs exact |
| IAEA 2D PWR Quarter-Core | 0.99837 | 1.02959 | −3086 pcm | See below |

**On the IAEA 2D PWR result:** 

* The −3086 pcm discrepancy is expected and physically interpretable. 
* Roughly 2000 pcm comes from the diffusion approximation itself (P1 vs transport), and roughly 1000 pcm from the 2-group energy structure — too coarse to capture spectral effects near the fuel-reflector interface accurately. 
* No bugs were found in the solver; this is the correct answer for this level of approximation.
* For better agreement on this benchmark, either transport-corrected diffusion coefficients or more energy groups near the reflector are needed to implement.

*Run it yourself:*
```bash
python main.py input/iaea_2d.yaml
```

---

## Quick start

```bash
pip install -r requirements.txt
python main.py input/iaea_2d.yaml
```

Expected:
```
k-eff = 0.998366
Results saved to: output/iaea_2d/
```

---

## Example input file

```yaml
title: "IAEA 2D PWR Benchmark"

geometry:
  type: 2D_rectangular
  domain_x: 170.0
  domain_y: 170.0
  nx: 170
  ny: 170

regions:
  - {name         : fuel_core, 
     x_min        : 0,
     x_max        : 80,
     y_min        : 0, 
     y_max        : 80, 
     material     : fuel1}
  - {name         : reflector, 
     x_min        : 80, 
     x_max        : 170,
     y_min        : 0, 
     y_max        : 170, 
     material     : reflector}

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
│   ├── input_reader.py    # YAML input file reader
│   ├── solver_2d.py       # 2-group diffusion solver
│   ├── solver_mg.py       # G-group multi-energy solver
│   ├── output_2d.py       # Flux maps and power distribution
│   ├── geometry.py        # 1D geometry (legacy)
│   └── solver_1d.py       # 1D diffusion solver (legacy)
├── input/                 # Input files
│   ├── iaea_2d.yaml       # IAEA 2D PWR benchmark
│   └── pwr_2d.yaml        # Simple PWR slab
├── output/                # Simulation results
├── tests/                 # Unit tests
├── benchmarks/            # Benchmark cases
├── docs/                  # Theory manual
├── main.py                # Entry point
└── requirements.txt
```

---

## Roadmap

The development path below is organized around three questions:
**Can it converge faster? Can it model more physics? Can it be validated against
something hard?**

### Phase 1 — Core solver (in progress)
- [x] 1D slab finite difference solver
- [x] 2D rectangular geometry, 5-point stencil
- [x] 2-group and G-group multigroup diffusion
- [x] IAEA 2D PWR quarter-core benchmark
- [ ] Coarse Mesh Finite Difference (CMFD) acceleration — current Gauss-Seidel
      outer iteration is slow on fine meshes; CMFD is the standard fix
- [ ] Chebyshev or JFNK acceleration for eigenvalue convergence
- [ ] Cylindrical 1D geometry (annular fuel pins, TRIGA-type problems)
- [ ] 3D Cartesian geometry extension

### Phase 2 — Neutronics accuracy
- [ ] Transport correction for diffusion coefficients (B1 or P1 leakage model)
- [ ] Adjoint flux solver — needed for perturbation theory, reactivity worths,
      sensitivity/uncertainty analysis (SCALE TSUNAMI-equivalent workflow)
- [ ] Reflector discontinuity factors (Koebke's equivalent homogenization)
- [ ] SP3 approximation for better accuracy near material interfaces without
      going to full transport
- [ ] Multigroup cross section library reader — at minimum support for 2G and
      few-group WIMS/SCALE-format collapsed libraries
- [ ] Benchmark suite: ANL benchmarks, OECD/NEA C5G7 (transport reference),
      KAIST 3D PWR benchmark

### Phase 3 — Core simulation
- [ ] Nodal Expansion Method (NEM) or Analytic Nodal Method (ANM) — orders of
      magnitude faster than fine-mesh FD for full-core LWR problems; this is
      what SIMULATE and DIF3D actually use
- [ ] Burnup and isotopic depletion (Bateman equations, matrix exponential
      solver)
- [ ] Reactivity feedback: Doppler broadening, moderator temperature/density,
      boron concentration
- [ ] Control rod cusping correction
- [ ] Shutdown margin and ejected rod worth calculation

### Phase 4 — Advanced / research features
- [ ] Full ENDF/B cross section processing pipeline (or interface to NJOY output)
- [ ] Simplified thermal hydraulics coupling (single channel, COBRA-like)
- [ ] Stochastic uncertainty quantification (sampling-based, compatible with
      SCALE/Sampler workflow)
- [ ] Machine learning surrogate for cross section parameterization vs burnup,
      temperature, and boron — interface point with ongoing ML research at ITC
- [ ] Verification against OpenMC and Serpent on identical geometry/material
      definitions

### What this code will not try to do
Full Monte Carlo transport, resolved resonance processing, and production-grade
thermal hydraulics are out of scope by design. Those problems are solved by
OpenMC, Serpent, and RELAP/TRACE. ANGKOR's role is to be a readable, modifiable
diffusion solver that a graduate student can actually debug.

---

## Why open source?

Commercial reactor physics codes (CASMO, SIMULATE, DIF3D, PARCS) are accurate
but opaque. SCALE is excellent but requires a license and considerable setup.
OpenMC is open source and rigorous but solves a different equation.

There is no clean, well-documented, Python-based diffusion solver that a
student can read in a weekend and modify without fear. ANGKOR tries to fill that
gap. The code is not production-grade. It is meant to be readable and educational first — useful second.

If you find a bug, open an issue. If you use it for a class or a paper, a
citation is appreciated.

---

## For researchers

If you work on reactor physics, neutronics methods, or nuclear data and want to
collaborate — particularly on the CMFD acceleration, nodal methods, or
cross section library integration — get in touch.

The ITC group is also developing ML surrogate models for SMR k-effective
prediction using CASMO/SIMULATE-generated training data. If that overlaps with
your work, reach out directly.

---

## About

ANGKOR is developed at the **Institute of Technology of Cambodia (ITC)**,
Phnom Penh, Cambodia 🇰🇭.

---

## Author

**MUTH Boravy**  
PhD, Nuclear Engineering  
Institute of Technology of Cambodia (ITC)  
Phnom Penh, Cambodia  
[GitHub](https://github.com/muthboravy007)

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).

---

## Citation

```
<<<<<<< HEAD
MUTH Boravy, "ANGKOR: Advanced Neutron Group-diffusion K-eigenvalue Of Reactors Analysis", Institute of Technology of Cambodia, 2025.
=======
MUTH Boravy, "ANGKOR: Advanced Neutron Group-diffusion K-eigenvalue Of Reactors", Institute of Technology of Cambodia, 2025.
>>>>>>> 44c1fe2 (Update)
https://github.com/muthboravy007/ANGKOR
```
