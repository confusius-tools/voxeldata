# VoxelData Agent Guidelines

This file guides AI coding agents working in this repository.

## What is VoxelData

VoxelData is a modality-independent model for spatially referenced
`xarray.DataArray` objects and a sister project of ConfUSIus. It owns generic geometry,
validation, extraction, and reconstruction—not probe hardware, acquisition
policies, slice-timing correction, or modality-specific analysis.

API details belong in `docs/api/` and the README; planned work belongs in
`ROADMAP.md`, not session logs.

## Data Model: VoxelData

- A **VoxelData array** has dimensions `(*nonspatial_dims, k, j, i)`. Native
  `(k, j, i)` map to world `(z, y, x)`; these names are fixed. Creation preserves
  leading dimension order; canonicalization uses `(...extra, time, k, j, i)`
  and restores scalar-selected voxel dimensions.
- A `VoxelToWorldIndex` owns lazily derived `z/y/x` coordinates. Ordinary indexes
  own native voxel and nonspatial labels; do not combine their ownership or
  materialize dense world coordinates unnecessarily. Affines act on native voxel
  labels, including after cropping, not newly zero-based positions.
- `geometry_dims` names ordered nonspatial affine dependencies. The shape is
  `(*[sizes[d] for d in geometry_dims], 4, 4)`; shared geometry uses `(4, 4)` and
  `geometry_dims=()`, regardless of nonspatial dimension names. Stacked affines
  require explicit dependencies; never infer them from names or matching lengths.
  Geometry may vary in scale.
- Timestamp dependencies are independent of geometry. A named time DataArray may
  depend on `time` plus any number of nonspatial axes; values must be numeric,
  finite, and increasing along `time` for every series. Select timestamp
  dependencies before shared-time indexing/analysis. Auxiliary coordinates such
  as `slice_time` remain consumer-defined; do not infer acquisition semantics.
- Use `create_voxeldata` to construct arrays (provide `spacing` or `voxel_to_world`),
  `validate_voxeldata` to check them, and `ensure_voxeldata` to canonicalize and
  validate. There is no plain-coordinate fallback for operations requiring
  VoxelData. Construction and `.vd.set_voxel_to_world` bootstrap plain arrays.

### Geometry and extraction contracts

- Keep the flat `.vd` accessor: no nested `.vd.affine` or `.voxeldata` alias.
  Mutators default to `inplace=False`; `inplace=True` returns the updated original.
  `.vd.origin`/`.vd.spacing` describe only spatial geometry (`z/y/x` world origin
  and `k/j/i` physical sample spacing). `.vd.dt` reports shared time sampling;
  select additional timestamp dependencies first. `.vd.world_units` describes
  world-coordinate units, not signal or time units. `.vd.bounding_box` and
  `.vd.geometry_center` return world-axis dictionaries enclosing all grids:
  `(min, max)` tuples and scalar centers, respectively. Select geometry
  dependencies first for grid-specific summaries.
- Named frames live on `VoxelToWorldIndex.named_affines`. Frames are finite, homogeneous
  world-frame `(4, 4)` matrices or full geometry-shaped stacks. Named frames need not
  share scales; index equality remains geometry-only.
- Applying a frame re-expresses other frames as `frame @ inv(action)`, aligned by
  dependency names, and stores an exact shared identity for the applied name.
  `set_voxel_to_world` drops frames with a warning unless explicitly supplied;
  unit changes and voxel reindexing preserve source frames, not reference frames.
- Spatial selection leaves frames unchanged; geometry selection slices their batch
  axes. Geometry-axis concatenation requires matching frame names: shared frames
  must agree, stacked frames concatenate, and mixing shared/stacked forms raises.
- Native `.reindex()`/alignment changing geometry labels is unsupported. Use
  `.isel()`/`.sel()`; `.vd` validation detects indexed-label mismatches, but raw
  world-coordinate access cannot. Do not monkey-patch Xarray to work around this.
  `.vd.reindex_voxels()` and `.vd.reindex_voxels_like()` are distinct, supported APIs.
  Select all geometry dependencies before a separate world-coordinate lookup.
- Binary extraction flattens all mask dimensions, including extra feature axes.
  Label maps are spatial-only, optionally stacked along `mask`. Shared spatial
  maps may span unmatched geometry dependencies only when physical grids agree;
  check native labels, affine geometry, and units, not merely array shapes.
- Base extraction must work without flox/Dask. Extras `accel` (flox) and `lazy`
  (Dask arrays) are independent; `all` combines runtime extras, not development
  tools. Preserve lazy signal data during extraction; masks/labels may be eager.
  Flox median may require explicit rechunking. `unmask` is eager and positional,
  preserving the mask's labels, geometry dependencies, units, and named frames.

## Commands

Use `uv` for dependencies/environments and `just` for common tasks.

```bash
uv sync
just test                 # pytest with coverage.
just pre-commit           # Ruff, ty, codespell, NumPy docstrings.
just docs                 # Strict Zensical documentation build.
just serve-docs
uv run ty check
uv build
```

Run one test: `uv run pytest tests/test_accessors.py::test_apply_named_frame_composition`.
CI configuration is in `.github/workflows/`; development dependencies include both
optional backends, so also verify base-only installs when changing dependencies.

## Code Architecture

- `src/voxeldata/geometry.py`: index/transform machinery, units, named frames.
- `affine.py`, `accessors.py`: geometry operations and the flat `.vd` interface.
- `create.py`, `validation.py`, `masks.py`: construction and model/grid validation.
- `extract/`: binary extraction, label reductions, and reconstruction.
- `_dims.py`, `_coordinates.py`: shared internal dimension/coordinate helpers.
- `__init__.py`: public exports and accessor registration; `tests/conftest.py`:
  shared function-scoped fixtures; `docs/api/`: API documentation.

### Module organization

- **Cross-module shared utilities** live in `voxeldata/_<topic>.py` with public
  function names (e.g. `voxeldata/_coordinates.py` exports
  `get_coordinate_spacing_info`, not `_get_coordinate_spacing_info`) — the leading
  `_` on the module conveys "internal API", names inside it are not prefixed.
  Group by topic; don't pile everything into one file.
- **Module-private shared helpers** live in `<module>/_utils.py` (e.g.
  `extract/_utils.py`) — for a helper shared by 2+ files inside one module but not
  used outside it.
- **Never import a `_name` across module boundaries.** Need a private function from
  another module? Either inline it, make it public in the same file, or promote it
  to a shared internal topic module. The underscore is a real boundary, not
  decoration.

## Coding Conventions

- **Imports**: absolute only (`from voxeldata import create_voxeldata`); group
  stdlib/third-party/local; use `TYPE_CHECKING` for type-only imports.
- **Naming**: `snake_case` functions/variables (prefer imperative verb phrases, e.g.
  `get_source_dataarray`, not noun phrases), `PascalCase` classes, `UPPER_CASE`
  constants, leading `_` for private.
- **Types**: comprehensive hints incl. `numpy.typing.NDArray`; `Literal` for string
  literals; `TypedDict` for structured dicts; `TypeAlias` for complex types; `py.typed`
  is enabled.
- **Errors**: specific exceptions (`ValueError`, `TypeError`, `FileNotFoundError`);
  `warnings.warn()` for non-critical issues; validate inputs early with descriptive
  messages.
- **Docstrings**: NumPy format for *all* functions/methods, public and private
  (private helpers still need full Parameters/Returns/Raises).
  - Optional params: `arg : type, optional` (default `None`) or
    `arg : type, default: value` — never `arg : type or None, default: None`.
  - Fallback behavior: "If not provided, ..." — never "If `None`, ...".
  - Booleans: start the description with "Whether to ..." — never "If `True`/`False`, ...".
  - Multiple return values: one `name : type` block per value in `Returns` — never
    `tuple[type1, type2]` as the documented return type.
  - Inline code: single backticks. Full package names in the `type` line
    (`xarray.DataArray`); no `xarray.` prefix/backticks in prose (`DataArray`).
  - Array shapes: `(X, Y, Z) numpy.ndarray` / `(X, Y, Z) xarray.DataArray`.
  - Cross-references: `[name][voxeldata.module.path.name]` — never Sphinx `.. [1]`.
    Only for objects rendered in the API docs. Private objects (`_name`, or anything
    under a `_module`/`_utils` package) are filtered out of the docs, so a `[][]` link
    to them fails the strict docs build; reference them as plain `` `name` `` instead.
  - Module-level constants get a triple-quoted docstring immediately after them.
- **Comments**:
  1. Comments should not duplicate code.
  2. Good comments do not excuse unclear code.
  3. If you can't write a clear comment, there may be a problem with the code.
  4. Comments should dispel confusion, not cause it.
  5. Explain unidiomatic code in comments.
  6. Provide links to the original source of copied code.
  7. Include links to external references where they will be most helpful.
  8. Add comments when fixing bugs.
  9. Use `TODO:` prefix for incomplete implementations.
  10. All comments end with a period.
  - Code adapted from another project (e.g. nilearn) needs a module-level `NOTICE`
    file reference: "Portions of this file are derived from [Project], which is
    licensed under the [License]. See `NOTICE` file for details."

## Test Conventions

- **No useless tests**: must fail if the function returns garbage — avoid tests that
  only check shape preservation or "output differs from input".
- **Test public API only**: don't test `_private` functions directly; they're covered
  via the public functions that use them.
- **No `# pragma: no cover`**, and don't force tests for branches that are unreachable
  due to an upstream library invariant.
- Cover: edge cases (empty inputs, boundaries), error validation (`pytest.raises`),
  and comparisons against reference implementations (scipy, naive implementations)
  where one exists. Reach for property-based tests only when no reference
  implementation exists (mathematical properties: idempotence, commutativity, ...).
- Use fixtures from `conftest.py` before creating new test data;
  `numpy.testing.assert_allclose`/`assert_array_equal`; seeded RNGs; small arrays.
- **VoxelData-specific coverage**: fresh function-scoped fixtures, independent
  numerical oracles and actual sample ordering, noncommuting frame composition,
  dependency slicing/order, native labels, frame preservation, Dask noncomputation,
  and native/accelerated label reductions. This package has no GUI/visual tests.

## Git & Changelog

Commit messages follow [Commitizen](https://commitizen.github.io/cz-cli/):

```
<type>(<scope>): <short summary>

<body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.

Scopes reflect this package: `geometry`/`affine`/`create`/`accessors`,
`extract`/`validation`, `docs`/`api`, `tests`.

VoxelData has no changelog yet. If introduced, use `docs/changelog.md` and the
following conventions:

`docs/changelog.md` is grouped by version, newest first. Add entries under the
current development version's `X.Y.Z.devN` heading (never a released version), under
section labels in this order (only add the ones you need):
`:boom: Breaking changes`, `:sparkles: Enhancements`, `:zap: Performance`,
`:bug: Fixes`, `:books: Documentation`, `:wrench: Maintenance`.
Write from the user's perspective (effect, not implementation), ending with a PR
link: `([#123](https://github.com/confusius-tools/voxeldata/pull/123))`.

## Releases

- Commit, push, merge, and publish only with authorization. Agree the release
  version explicitly. Merge and verify changes on `main` before tagging a release;
  do not release from an unmerged feature branch.
- Update `project.version` and `uv.lock`, then tag the tested `main` commit with
  matching `vVERSION`. `.github/workflows/publish.yml` gates PyPI Trusted Publishing
  on lint, the OS/Python test matrix, docs, build, and the `pypi` environment approval.
- Never move published tags or replace published artifacts.

Keep publishing setup out of the README; inspect the workflow rather than
duplicating setup here.
