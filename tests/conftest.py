import random

import pytest

from amlearn.domains import BENCHMARKS, Benchmark


@pytest.fixture(params=sorted(BENCHMARKS))
def bench(request) -> Benchmark:
    return BENCHMARKS[request.param]


@pytest.fixture
def blocks() -> Benchmark:
    return BENCHMARKS["blocksworld"]


@pytest.fixture
def rng() -> random.Random:
    return random.Random("tests")
