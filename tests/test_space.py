import math

import numpy as np
import pytest

from molopt import Choice, Constant, FiniteSpace, Integer, Real, Space, ValidationError


def test_product_validation_and_constants():
    space = Space(
        temperature=Real(300, 800),
        concentration=Real(0.001, 1.0, log=True),
        chain=Integer(10, 100, step=5),
        architecture=Choice(["linear", "block"]),
        solvent=Constant("water"),
    )
    params = space.validate(
        {
            "temperature": 300,
            "concentration": 0.01,
            "chain": 15,
            "architecture": "block",
        }
    )
    assert params["solvent"] == "water"
    assert space.identity(params) == space.identity({**params, "solvent": "water"})
    assert not space.contains({"temperature": 299, "concentration": 0.01, "chain": 15, "architecture": "block"})
    with pytest.raises(ValidationError):
        space.validate({**params, "extra": 1})


def test_log_changes_measure_not_integer_grid():
    variable = Integer(2, 8, log=True, step=2)
    assert list(variable.enumerate()) == [2, 4, 6, 8]
    rng = np.random.default_rng(0)
    for _ in range(30):
        assert variable.sample(rng) in {2, 4, 6, 8}


def test_real_step_grid_and_log_conflict():
    variable = Real(0.0, 1.0, step=0.5)
    assert list(variable.enumerate()) == [0.0, 0.5, 1.0]
    assert variable.validate(0.5 + 1e-12) == pytest.approx(0.5)
    with pytest.raises(ValidationError):
        Real(0.1, 1.0, log=True, step=0.1)
    with pytest.raises(ValidationError):
        Real(-1.0, 1.0, log=True)


def test_choice_keeps_bool_distinct_from_int():
    space = Space(flag=Choice([True, 1]))
    assert space.contains({"flag": True})
    assert space.contains({"flag": 1})
    assert space.identity({"flag": True}) != space.identity({"flag": 1})
    with pytest.raises(ValidationError):
        Choice(["a", "a"])
    flags = Space(flag=Choice([True, False]))
    assert flags.contains({"flag": True})
    assert not flags.contains({"flag": 1})


def test_enumeration_order_and_cardinality():
    space = Space(a=Integer(0, 1), b=Choice(["x", "y"]), c=Constant(1))
    assert space.cardinality() == 4
    assert list(space.enumerate()) == [
        {"a": 0, "b": "x", "c": 1},
        {"a": 0, "b": "y", "c": 1},
        {"a": 1, "b": "x", "c": 1},
        {"a": 1, "b": "y", "c": 1},
    ]
    assert Space(x=Real(0, 1)).cardinality() is None
    with pytest.raises(ValidationError):
        list(Space(x=Real(0, 1)).enumerate())


def test_finite_space_identity_and_rejection():
    space = FiniteSpace(
        [
            {"fraction": 0.2, "architecture": "block"},
            {"fraction": 0.5, "architecture": "random"},
        ]
    )
    assert space.cardinality() == 2
    with pytest.raises(ValidationError):
        FiniteSpace(
            [
                {"fraction": 0.2, "architecture": "block"},
                {"fraction": 0.2, "architecture": "block"},
            ]
        )
    with pytest.raises(ValidationError):
        space.validate({"fraction": 0.2, "architecture": "gradient"})
    rng = np.random.default_rng(1)
    for params in space.sample(20, rng):
        assert space.contains(params)
    with pytest.raises(AttributeError):
        space.variables


def test_sampling_stays_inside_mixed_space():
    space = Space(
        x=Real(0.1, 1.0, log=True),
        n=Integer(1, 5),
        kind=Choice(["a", "b", "c"]),
        fixed=Constant(3),
    )
    rng = np.random.default_rng(2)
    for params in space.sample(40, rng):
        assert space.contains(params)
        assert params["fixed"] == 3
        assert params["x"] > 0
        assert math.isfinite(params["x"])
