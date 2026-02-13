"""Short experiment on random 5x5 zero-sum games."""
import numpy as np
from experiments import run_suite


def main():
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, size=(5, 5))
    run_suite(A, T=1500, seeds=10)


if __name__ == "__main__":
    main()
