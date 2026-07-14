# Test Suite for Masters Codebase

The tests are implemented using pytest and are organised to mirror the structure of the source code.

## Structure

The test suite is organized as follows:

```
tests/
├── analysis/           # Tests for analytical tools
├── protocols/          # Tests for protocols
├── conftest.py         # Common fixtures and test utilities
└── README.md           # This file
```

## Running Tests

To run the tests, use the following command from the project root:

```bash
pytest
```

To run tests with coverage reporting:

```bash
pytest --cov=src
```

## Coverage Goals

The goal is to achieve at least 50% test coverage for the codebase, with a focus on:
1. Analytical tools in `src/analysis/`
2. Protocol implementations in `src/protocols/`

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_spike_data`: Provides a sample DataFrame with spike data for testing
- `sample_config`: Provides a sample configuration dictionary
- `sample_paths`: Provides sample paths for raw and results data
- `ensure_test_dirs`: Ensures test directories exist

## Writing Tests

When writing tests, follow these guidelines:

1. Create test files with the naming pattern `test_*.py`
2. Create test functions with the naming pattern `test_*`
3. Use fixtures from `conftest.py` where appropriate
4. Include docstrings for all test functions
5. Test both normal and edge cases (empty data, invalid inputs, etc.)
6. Use mocks for external dependencies when appropriate

## Test Categories

The tests are categorized as follows:

### Analytical Tools Tests

Tests for the analytical tools `src/analysis/`. Each tool has its own test file:

- `test_fft.py`: Tests for FFT analysis
- `test_isi.py`: Tests for Inter-Spike Interval analysis
- `test_correlation.py`: Tests for correlation analysis
- etc.

### Protocol Tests

Tests for the protocol implementations `src/protocols/`. Each protocol has its own test file:

- `test_base.py`: Tests for the base protocol class
- `test_baseline.py`: Tests for the baseline protocol
- etc.