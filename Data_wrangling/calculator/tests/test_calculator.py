from gerardocalc.calculator import Calculator
import py.test
import itertools
from hypothesis import given, assume, strategies as st
from math import isclose, isnan
import cmath


def test_calc():
    calc = Calculator()
    args = [10, 1, 18, -6, 3, 0.666, 8.4, -9, -0.1, 0.3]
    for t in itertools.permutations(args, 2):
        x, y = t
        calc.state = y
        x1 = calc.add(x)
        assert x1 == (x + y)
        y = calc.state
        x2 = calc.subs(x)
        assert x2 == (y - x)
        y = calc.state
        x3 = calc.mul(x)
        assert x3 == (y * x)
        y = calc.state
        x4 = calc.div(x)
        assert x4 == (y / x)
        y = calc.state
        x5 = calc.root(x)
        assert x5 == (y ** (1 / x))


def test_types():
    with py.test.raises(TypeError):
        calc = Calculator()
        x = "g"
        y = 0
        calc.state = y
        x1 = calc.add(x)
        assert x1 == (x + y)
        y = calc.state
        x2 = calc.subs(x)
        assert x2 == (y - x)
        y = calc.state
        x3 = calc.mul(x)
        assert x3 == (y * x)
        y = calc.state
        x4 = calc.div(x)
        assert x4 == (y / x)
        y = calc.state
        x5 = calc.root(x)
        assert x5 == (y ** (1 / x))
    with py.test.raises(TypeError):
        calc = Calculator()
        x = "5"
        y = 0
        calc.state = y
        x1 = calc.add(x, y)
        assert x1 == (x + y)
        y = calc.state
        x2 = calc.subs(x, y)
        assert x2 == (y - x)
        y = calc.state
        x3 = calc.mul(x, y)
        assert x3 == (y * x)
        y = calc.state
        x4 = calc.div(x, y)
        assert x4 == (y / x)
        y = calc.state
        x5 = calc.root(x, y)
        assert x5 == (y ** (1 / x))


@given(
    st.floats(min_value=-5e3, max_value=5e3), st.floats(min_value=-5e3, max_value=5e3)
)
def test_hypo(x: float, y: float):
    assume(abs(x) > 0)
    assume(not isnan(x))
    assume(not isnan(y))
    calc = Calculator()
    calc.state = y
    x1 = calc.add(x)
    assert isclose(x1, (x + y), abs_tol=1e-5)
    calc.state = y
    x2 = calc.subs(x)
    assert isclose(x2, (y - x), abs_tol=1e-5)
    calc.state = y
    x3 = calc.mul(x)
    assert isclose(x3, (y * x), abs_tol=1e-5)
    calc.state = y
    x4 = calc.div(x)
    assert isclose(x4, (y / x), abs_tol=1e-5)
    calc.state = y
    if (abs(x) > 0.05) and (
        abs(y) > 0
    ):  # To exclude overflow and division by 0, both are handled
        x5 = calc.root(x)
        assert cmath.isclose(x5, (y ** (1 / x)), abs_tol=1e-5)
