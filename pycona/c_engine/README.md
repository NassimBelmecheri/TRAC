# pycona C++ acquisition engine

A self-contained, high-speed **native** constraint-acquisition engine used by
`pycona/run_benchmark.py`. It implements the interactive acquisition algorithms
(QuAcq, PQuAcq, MQuAcq, MQuAcq2, GrowAcq, BruteCA) plus the passive ConAcq, a
lightweight MAC/AC-3 CP solver, FindScope (v1/v2) and FindC (v1/v2) — all in
plain C++17 with **no third-party dependencies** — and exposes them to Python
through a small `ctypes` layer (`c_interface.py`, `adapter.py`).

It is independent of the neural TRAC platform (`trac_platform/`): that pipeline
uses the pure-Python `pycona` learner with the transformer oracle. This engine
is the fast, symbolic-only alternative for the classical benchmarks.

```
c_engine/
  c_api.cpp / c_api.h      C ABI exported from the shared library (extern "C")
  quacq.cpp / quacq.h      the acquisition algorithms + FindScope/FindC
  solver.cpp / solver.h    MAC/AC-3 CP solver (query generation, split search)
  constraint.h domain.h constraint_net.h common.h   core data structures
  c_interface.py           ctypes bindings (CQuAcqEngine)
  benchmarks.py            native benchmark builders (zebra, sudoku, ...)
  build.py                 cross-platform build helper  (recommended)
  Makefile                 make-based build (Linux/macOS/MinGW)
  verify.py                acquisition-correctness checker (native solver, no CPMpy)
  solver_check.py          solver reliability check vs brute-force ground truth
```

Everything here is **CPMpy-free**: benchmarks are built directly on the native
engine and the correctness tooling uses the engine's own C++ solver, so the only
dependency is the compiled library (plus the Python standard library).

The Python loader (`c_interface.py`) expects the compiled library next to the
sources: **`c_engine.dll`** on Windows, **`c_engine.so`** on Linux/macOS. If it
is missing you will get:

```
FileNotFoundError: Could not find compiled library at .../c_engine.dll.
Please compile c_engine.
```

A prebuilt `c_engine.dll` (Windows x64) is checked in. **Rebuild whenever you
edit the C++ sources, or when you are on Linux/macOS** (the checked-in `.dll`
is Windows-only).

---

## 1. Prerequisites

Any C++17 compiler. No libraries, no CMake required.

| Platform | Install a compiler | Provides |
|----------|--------------------|----------|
| **Windows** | [MSYS2 / MinGW-w64](https://www.msys2.org/) (`pacman -S mingw-w64-ucrt-x86_64-gcc`), or the **Visual Studio Build Tools** (MSVC `cl`). MinGW also ships with [Strawberry Perl](https://strawberryperl.com/). | `g++` or `cl` |
| **Linux** | `sudo apt install build-essential` (Debian/Ubuntu) or `sudo dnf install gcc-c++` | `g++` |
| **macOS** | `xcode-select --install` | `clang++` |

Check it is on your `PATH`:

```bash
g++ --version        # or: clang++ --version   /   cl
```

---

## 2. Build

### Option A — cross-platform helper (recommended)

Auto-detects the compiler and produces the correctly-named library in place.

```bash
cd pycona/c_engine
python build.py --verify
```

`--verify` loads the freshly built library through `ctypes` and creates/frees an
engine to prove it works. Useful flags:

```bash
python build.py                 # build with the auto-detected compiler
python build.py --compiler cl   # force MSVC (or g++, clang++)
python build.py --debug         # -O0 -g (or /Od /Zi) build with symbols
```

On Windows the GCC/Clang builds are linked **statically** against
libstdc++/libgcc, so the resulting DLL has no runtime dependency on the MinGW
runtime DLLs and loads inside any Python process.

### Option B — make (Linux / macOS; MinGW with an explicit compiler)

```bash
cd pycona/c_engine
make            # optimized -> c_engine.so (or c_engine.dll on Windows)
make debug      # unoptimized with symbols
make clean
```

On Windows, `make` is only convenient under MSYS2. Some standalone MinGW
distributions (e.g. Strawberry Perl's `mingw32-make`) ship a broken built-in
`CXX` and no `sh.exe`; pass the compiler explicitly, e.g.
`mingw32-make CXX=g++`, or just use **`build.py`** (Option A), which is the
recommended and tested path on Windows.

### Option C — manual one-liners

```bash
# Linux / macOS
g++ -std=c++17 -O3 -shared -fPIC -o c_engine.so c_api.cpp quacq.cpp solver.cpp

# Windows, MinGW g++ (self-contained DLL)
g++ -std=c++17 -O3 -shared -static -static-libgcc -static-libstdc++ \
    -o c_engine.dll c_api.cpp quacq.cpp solver.cpp

# Windows, MSVC (from a "x64 Native Tools" prompt)
cl /nologo /std:c++17 /EHsc /O2 /LD c_api.cpp quacq.cpp solver.cpp /Fe:c_engine.dll
```

### Verify the build loads

```bash
cd pycona
python -c "from c_engine import CQuAcqEngine; e=CQuAcqEngine(9); print('engine OK')"
```

---

## 3. Running acquisition

```bash
cd pycona
python run_benchmark.py zebra --algo all          # pquacq, quacq, growacq, bruteca
python run_benchmark.py sudoku4 --algo quacq
python run_benchmark.py nqueens -n 8 --findc2      # non-normalised -> use FindC2
```

From Python:

```python
from c_engine import CQuAcqEngine, OP_NE
eng = CQuAcqEngine(9)
for i in range(9):
    eng.set_domain(i, [1,2,3,4,5,6,7,8,9])
# ... add target/bias constraints ...
eng.set_findc_version(2)          # see note below
learned = eng.run("quacq")
print(eng.get_metrics())
```

### Normalised vs non-normalised networks — pick the right FindC

* **Normalised** networks have **at most one constraint per scope** (e.g. the
  AllDifferent `!=` cliques of Sudoku/Latin squares). The classic **FindC (v1)**
  is correct and query-efficient here — this is the default.
* **Non-normalised** networks put **several constraints on one scope** — e.g.
  n-queens has `!=` **and** the two diagonal `x-y!=±(i-j)` on each pair; the
  zebra puzzle mixes `!=`, `|x-y|==1` and unary clues; golomb rulers add
  ordering + all-different distances. FindC v1 can only return one constraint
  per scope, so it under- or over-constrains. **Use `--findc2` /
  `set_findc_version(2)` for these** — it learns the full conjunction on each
  scope.

---

## 4. Checking correctness

`run_benchmark.py` only reports whether the learned network is *satisfiable*
("SAT?"), which does **not** prove it was acquired correctly (an
under-constrained network is still SAT). Use `verify.py` to check **solution-set
equivalence** between the learned network `C_L` and the target `C_T`. It uses the
engine's **own native C++ solver** — no CPMpy, no OR-Tools — asking two
questions with `SOLVE_VIOLATE_BIAS`:

* is there an assignment satisfying `C_L` that violates `C_T`?  → **TOO_WEAK**
* is there a genuine `C_T` solution that violates `C_L`?        → **TOO_STRONG**

Only the constraints that *differ* between `C_L` and `C_T` are handed to the
solver, so the common cases are one quick solve (or none).

```bash
cd pycona
python c_engine/verify.py                              # all benchmarks (auto FindC)
python c_engine/verify.py -b zebra nqueens8 -a quacq bruteca
python c_engine/verify.py --findc both                 # compare FindC v1 vs v2
python c_engine/verify.py --timeout 20                 # per-solve time limit
```

`--findc auto` (default) uses **v1 for the normalised benchmarks (sudoku)** and
**v2 for the rest**. Each run is classified as:

| Status | Meaning |
|--------|---------|
| `EXACT` | learned set == target set |
| `EQUIV` | differs syntactically but same solution set (extra constraints are redundant / entailed) |
| `TOO_WEAK` | learned network admits a non-solution of the target |
| `TOO_STRONG` | learned network rejects a genuine target solution |
| `INCONCLUSIVE` | no counter-example found, but the native solver hit its time limit while *proving* entailment (a hard co-NP proof — see the sudoku 9×9 note below) |

The checker is **sound**: finding a counter-example (TOO_WEAK / TOO_STRONG) is a
fast SAT search, so a genuine defect is never mislabelled as correct. It never
trusts the engine's own `converged` flag as a proof — that is exactly what it is
there to double-check.

### Is the native solver trustworthy? (`solver_check.py`)

Because the equivalence check relies on the C++ solver, `solver_check.py`
independently validates that solver against a **pure-Python brute-force ground
truth** (no CPMpy): it enumerates hundreds of small random networks plus every
benchmark target and confirms the solver's SAT/UNSAT verdict, that any returned
assignment is genuinely valid, and that `SOLVE_VIOLATE_BIAS` really violates the
bias.

```bash
cd pycona
python c_engine/solver_check.py            # benchmarks + 400 random nets
python c_engine/solver_check.py -n 3000     # more random cases
```

Result on the current build: **RELIABLE** — 3000+ `SOLVE_ANY` /
`SOLVE_VIOLATE_BIAS` checks with **0 mismatches**, all benchmark targets solve to
valid solutions, UNSAT is distinguished from TIMEOUT (`CQuAcqEngine.solve_ex`
returns the raw `SAT`/`UNSAT`/`TIMEOUT` status), the timeout is honored to within
a fraction of a second, and acquisition is deterministic across runs.

---

## 5. Verified correctness status

With `--findc auto`, `verify.py` reports **EXACT/EQUIV for every algorithm** on:

| Benchmark | Type | FindC | quacq | pquacq | mquacq | mquacq2 | growacq | bruteca |
|-----------|------|:----:|:----:|:-----:|:-----:|:------:|:------:|:------:|
| sudoku 4×4        | normalised | v1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| zebra             | non-norm.  | v2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| n-queens (4, 8)   | non-norm.  | v2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

(“✅” = EXACT or EQUIV, converged.)

`bruteca` (the FASTCA algorithm) additionally acquires the **golomb** ruler
(4- and 8-mark) EXACT/EQUIV — including the quaternary all-different-distance
constraints — so it acquires **all seven benchmarks**:

| Benchmark | FindC | bruteca | queries |
|-----------|:----:|:------:|:------:|
| sudoku 4×4 | v1 | ✅ EXACT | 449 |
| sudoku 9×9 | v1 | ✅ EXACT | 11 809 |
| zebra      | v2 | ✅ EQUIV | 2 429 |
| n-queens 4 | v2 | ✅ EQUIV | 112 |
| n-queens 8 | v2 | ✅ EQUIV | 1 900 |
| golomb 4   | v2 | ✅ EQUIV | ~10 000 |
| golomb 8   | v2 | ✅ EQUIV | ~125 000 |

### sudoku 9×9 — correct but hard to *prove* equivalent

`bruteca` acquires sudoku 9×9 **EXACT** (810/810). The other algorithms converge
having added **nothing wrong** (`extra = 0`) but learn a smaller, equivalent
`≠`-network (e.g. 648 constraints); proving that the ~160 omitted constraints are
redundant is a hard co-NP query, so `verify.py` reports `INCONCLUSIVE` on those
(any solver — including CPMpy/OR-Tools — times out here). They are equivalent in
principle; only the *proof* is expensive.

### golomb ruler — high query cost for the quaternary constraints

The **quaternary** all-different-distance constraints `|x_i−x_j| ≠ |x_k−x_l|`
(scope 4 over domain 1..35) cannot be enumerated exhaustively, so FindC v2
confirms them by **random sampling** on the scope: a candidate is real iff no
oracle-accepted sample violates it (see §6). This makes `bruteca` acquire golomb
exactly, but it is query-intensive (golomb-8 needs ~125 000 queries — pass
`-m 500000` to `verify.py`).

The query-generation algorithms (quacq / pquacq / mquacq / mquacq2 / growacq) do
**not** acquire golomb: they drive acquisition through `generate_query →
find_scope → find_c`, and on golomb `find_scope` cannot isolate the many
overlapping quaternary scopes, so `find_c` confirms nothing and the bias stops
shrinking (a **livelock** — the learned count stays flat while queries are
consumed up to the budget). They therefore report `TOO_WEAK` on golomb. Making
those algorithms acquire golomb needs a `find_scope` that handles many
overlapping high-arity scopes; `bruteca` (the FASTCA algorithm, and the one the
neural TRAC platform uses) side-steps this by enumerating candidate scopes
directly, which is why it acquires all seven benchmarks. Everything except golomb
is acquired exactly by **every** algorithm.

---

## 6. Implementation notes / recent fixes

The engine received several correctness fixes (all covered by `verify.py`):

* **FindC v1 termination.** The discriminator loop used to fall back to
  `SOLVE_ANY` when the split search was UNSAT, producing a non-splitting example
  that never shrank the candidate set — an **infinite loop** (not bounded by the
  query budget) on non-normalised networks. It now stops and returns a survivor.
* **FindC v2 (conjunctions).** Sound *FindAllC*: scopes small enough to
  enumerate are handled **exhaustively** (a candidate is confirmed iff every
  oracle-accepted assignment on the scope satisfies it) — sound and complete,
  fixes n-queens/zebra. Scopes too large to enumerate (golomb's quaternary
  distance constraints over domain 1..35) use the **sampled** analogue: draw
  assignments respecting the learned context, and confirm a candidate iff no
  oracle-accepted sample violates it and some rejected sample does. This never
  confirms a spurious order-relation (`G<`/`G>`/`G==`) on a distance pairing —
  the earlier split-half heuristic could not separate the ~60 candidates sharing
  one 4-variable scope and produced both `TOO_STRONG` and `TOO_WEAK` networks.
* **MQuAcq / MQuAcq2.** MQuAcq no longer over-removes the bias (it previously
  "converged" after learning a single constraint); MQuAcq2 no longer spins
  forever (`count_rejects` was being passed a variable-index list instead of an
  assignment).
* **Convergence flag.** GrowAcq and BruteCA now set `converged` when they
  exhaust their candidates within the query budget.

Everything is pure standard C++17 — after editing any `.cpp`/`.h`, just rerun
`python build.py --verify`.
