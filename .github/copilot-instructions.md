# GitHub Copilot Instructions for `hamilflow`

Always follow these rules when developing in the `hamilflow` repository.

## Core Philosophical Principles
- **Physical Accuracy**: Mathematical formulas must be derived from first principles and cited where possible.
- **Performance**: Use vectorised NumPy/SciPy operations instead of Python loops.
- **Type Safety**: Use strict type hinting for all public and private methods.

## Environment & Dependency Management
- **Tooling**: Always use `uv` for package management and script execution.
- **Commands**:
  - Add dependency: `uv add <package>`
  - Run tests: `uv run pytest`
  - Lint: `uv run ruff check .`
- **pre-commit**: run pre-commit to check the code style and formatting.

## Python Style & Standards
- **Formatter/Linter**: Follow Ruff/Black defaults (line-length = 88).
- **Imports**:
  - Grouping: Standard library, third-party, first-party (`hamilflow`).
  - Style: Use `from x.y import z` for first-party modules.
- **Architecture**:
  - **Models**: Inherit from `pydantic.BaseModel` for parameter sets and initial conditions.
  - **Logic**: Use `abc.ABC` and `abstractmethod` for base physical models.
  - **Properties**: Use `@cached_property` and `@computed_field` for derived values.
- **Typing**:
  - Use `numpy.typing.ArrayLike` (aliased as `npt.ArrayLike`) for arrays.
  - Use `collections.abc.Sequence` or `Mapping` for generic containers.
  - Use `TYPE_CHECKING` for complex circular imports in type hints.

## Documentation (Docstrings)
- **Style**: Use a mix of reStructuredText tags (`:param`, `:return`, `:cvar`) and Markdown.
- **Init**: Do NOT document `__init__` methods; document the class docstring instead.
- **Mathematics**:
  - Use LaTeX for all mathematical formulas.
  - Inline: `$ formula $`
  - Block: `$$ formula $$`
- **Citations**: Use Wikipedia or academic references in Markdown links or footers.

## Testing Standards
- **Framework**: `pytest`.
- **Location**: Mirror the package structure under `tests/`.
- **Patterns**:
  - Use `@pytest.fixture` for common setups.
  - Use `@pytest.mark.parametrize` for numerical verification across inputs.
  - Use `pd.testing.assert_frame_equal` for DataFrame comparisons.
  - Use `np.testing.assert_array_equal` or `assert_allclose` for numerical arrays.

## Git & Commits
- Use commitizen to commit.
