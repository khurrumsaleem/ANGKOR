# `test_mg_cmfd.py` Debug + Physics/Numerics Summary

Date: 2026-04-02  
Workspace: `D:\1.NUCLEAR\5_ANGKOR`

---

## 1) What was checked

I scanned the project structure and focused on:

- `D:\1.NUCLEAR\5_ANGKOR\tests\test_mg_cmfd.py`
- `D:\1.NUCLEAR\5_ANGKOR\test_mg_cmfd.py`
- `D:\1.NUCLEAR\5_ANGKOR\angkor\solver_mg.py`
- `D:\1.NUCLEAR\5_ANGKOR\angkor\cmfd.py`
- `D:\1.NUCLEAR\5_ANGKOR\angkor\input_reader.py`
- `D:\1.NUCLEAR\5_ANGKOR\input\pwr_2d.yaml`

---

## 2) Immediate failing error in `tests/test_mg_cmfd.py`

### Error

`TypeError: SolverMG.__init__() missing 1 required positional argument: 'n_groups'`

### Cause

In `tests/test_mg_cmfd.py`, `SolverMG(...)` is called without `n_groups`, but constructor is:

```python
SolverMG(engine, materials, settings, n_groups, boundary=None)
```

---

## 3) Additional structural issues

1. `tests/test_mg_cmfd.py` is script-like (runs on import), not proper pytest style.
2. `angkor/input_reader.py` imports `geometry_2d` as top-level module, which can fail under package imports.
3. There are **two similarly named files**:
   - `D:\1.NUCLEAR\5_ANGKOR\test_mg_cmfd.py` (root script)
   - `D:\1.NUCLEAR\5_ANGKOR\tests\test_mg_cmfd.py` (pytest file)

---

## 4) Physics / numerical findings (important)

### Baseline (no CMFD)

- `k ≈ 1.179617709`
- converges in ~74 iterations
- eigen residual is small (`~3.5e-7`) -> physically consistent

### Current MG-CMFD behavior

- `k ≈ 1.179870221`
- difference vs baseline: **~25.25 pcm**
- converges in ~48 iterations
- eigen residual is large (`~4.17e-2`) -> not truly settled fine-mesh eigenstate

### Root numerical reason

CMFD is applied too often and too late into final convergence, so coarse-mesh correction bias contaminates final `k`.

---

## 5) Step-by-step changes you should do

## Step A: fix constructor usage in test file

In `D:\1.NUCLEAR\5_ANGKOR\tests\test_mg_cmfd.py`, add:

- `n_groups=2`
- `boundary=reader.boundary`

for both `SolverMG(...)` calls.

---

## Step B: make test a real pytest test

Refactor file into at least one `def test_...():` function and add assertions, e.g.:

- `diff_pcm < 5.0`
- optional eigen residual threshold

Do not leave solver runs at module top-level.

---

## Step C: fix package import robustness

In `D:\1.NUCLEAR\5_ANGKOR\angkor\input_reader.py`, change import to:

```python
try:
    from .geometry_2d import Region, GeometryEngine
except ImportError:
    from geometry_2d import Region, GeometryEngine
```

---

## Step D: tune MG-CMFD gating in `solver_mg.py`

In `solve()` of `D:\1.NUCLEAR\5_ANGKOR\angkor\solver_mg.py`:

1. Actually use `cmfd_interval`.
2. Use a two-sided CMFD window:
   - upper gate: `k_error < 5e-3`
   - lower gate: `k_error > 5e-5`
3. Add shape gate: `phi_error < 1e-2`
4. Delay first CMFD application until at least iteration 10.

Recommended condition pattern:

```python
if cmfd_active and (iteration + 1) % cmfd_interval == 0 \
   and iteration >= 9 and k_error < 5e-3 and k_error > 5e-5 \
   and phi_error < 1e-2:
    ...
```

---

## Step E (recommended): disable CMFD near final convergence

When `k_error <= 5e-5`, stop applying CMFD and finish with plain power iterations.

This avoids freezing coarse-bias near the end.

---

## Step F: convergence check choice

For accuracy, prefer requiring both `k_error` and `phi_error` at final stage (or after CMFD is disabled), not only `k_error`.

---

## 6) Validation commands

Run after your edits:

```powershell
.\venv\Scripts\python.exe -m pytest -q tests/test_mg_cmfd.py
```

Then benchmark script:

```powershell
.\venv\Scripts\python.exe test_mg_cmfd.py
```

Expected direction:

- constructor/import error gone
- CMFD `k` bias drops from ~25 pcm to within test tolerance target
- residual stays small near convergence

---

## 7) Note about other tests

`D:\1.NUCLEAR\5_ANGKOR\tests\test_cmfd_mg.py` currently fails for independent reasons:

1. It expects per-group total preservation in `rebalance_flux_mg`, but current implementation intentionally changed this behavior.
2. It uses `xs["sigma_s"][g, g2]` assumptions that mismatch list-vs-array in stub data.

So full-suite stability may require updating that test file too.

---

## 8) Short conclusion

The immediate failure is a missing `n_groups` argument in test setup.  
After that, the real issue is numerical: MG-CMFD gating strategy is too aggressive and causes final eigenvalue bias.  
Apply CMFD only in a controlled mid-convergence window and stop it near final convergence.

