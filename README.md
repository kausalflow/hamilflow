# hamilflow

Dataset of simple physical systems.

## Publish process
- Squash merge all features to a branch, e.g. `main` or `release/cmp0xff/candidate`
    - Use `feat(section): xxx`, `fix(section): yyy`, `chore(poetry): lock` etc as the squashed commit messages
- Run `cz bump --dry` to determine the new version `X.Y.Z`. Or check out the [documentation](https://commitizen-tools.github.io/commitizen/bump/) for imposing a version
- Run `git checkout release/X.Y.Z` (-b? not sure... I have always been using SourceTree )
- Run `cz bump`. This will
    - Write `CHANGELOG.md` from the commit messages; update the version in `pyproject.toml`
    - Commit `CHANGELOG.md` and `pyproject.toml`
    - Create a new tag
- Push to `release/X.Y.Z`
- Merge to `main`, no need to squash, maybe don't delete the branch under release
