# MolOpt

Ask–tell black-box optimization. One problem definition works with every algorithm. The caller evaluates candidates.

MolOpt owns the search space, objectives, outcome constraints, trials, results, and optimizer state. It does not run experiments, store an experiment database, or read molecular file formats.

## Install

The core depends only on NumPy and SciPy. Each algorithm imports on its own, so an algorithm does not require a numerical library it does not use.

```bash
pip install molopt
```

Bayesian optimization calls BoTorch directly. Install that extra when you use it. There is no switchable backend. A later CMA-ES would use pycma the same way. Optuna, pymoo, and scikit-learn are not runtime dependencies, and MolOpt does not ship a second Gaussian process.

```bash
pip install "molopt[bo]"
```

MolOpt owns the search space, objectives, constraints, the ask–tell loop, optimizer state, and the choice of what to propose. BoTorch supplies Gaussian-process inference and acquisition values. Algorithms that do not need PyTorch never import it.

## Ask and tell

```python
from molopt import Constraint, Maximize, Problem, Real, Choice, Result, Space
from molopt.algorithms import BayesianOptimization, NSGA2

problem = Problem(
    space=Space(
        fraction=Real(0.1, 0.9),
        architecture=Choice(["random", "gradient", "block"]),
    ),
    objectives=[Maximize("conductivity"), Maximize("modulus")],
    constraints=[Constraint("stability", lower=0.8)],
)

# pip install "molopt[bo]"
optimizer = BayesianOptimization(problem, seed=42, n_initial=6, noise=0.0)
# optimizer = NSGA2(problem, seed=42, population_size=32)

trials = optimizer.ask(n=4)
for trial in trials:
    optimizer.tell(Result(trial_id=trial.id, values=evaluate(trial.params)))

front = optimizer.pareto_front()
```

`evaluate` belongs to the caller. `best()` and `pareto_front()` return observed feasible measurements. A model prediction is not a result.

`noise=0.0` is an explicit noiseless assertion. Omitting both `noise` and `stderr` is an error for Bayesian optimization, because missing uncertainty is not treated as zero.

## Algorithms in v0.1

| Algorithm | Where it applies |
|---|---|
| `RandomSearch` | Any supported space |
| `GridSearch` | Finite spaces up to `max_enumerate` |
| `LatinHypercube` | Continuous `Real` bounds, via SciPy QMC |
| `NSGA2` | Product spaces and finite libraries. Outcome constraints use Deb's rule, not a penalty |
| `BayesianOptimization` | Continuous boxes, finite libraries, and small mixed spaces. Requires `molopt[bo]` |

An unsupported problem raises `CapabilityError` when the optimizer is constructed.

## What stays outside

MolOpt does not import MolNex, Molab, MolPy, or MolQ. Checkpointing restores optimizer state. It is not an experiment log.

`examples/polymer_electrolyte.py` compares random search, NSGA-II, and Bayesian optimization on one finite candidate library and one synthetic oracle.
