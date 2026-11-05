# AGENTS.md — voxeldata

This file lets a new agent resume work on the standalone **VoxelData** package.

## Objective

Extract ConfUSIus's VoxelData model into an independent, modality-independent
package at:

```
/home/sdiebolt/Documents/Work/cortexlab/confusius-tools/voxeldata
```

Source project: `/home/sdiebolt/Documents/Work/cortexlab/confusius-tools/confusius`.
Downstream consumer: `/home/sdiebolt/Documents/Work/cortexlab/registration/voxreg`.

The initial package includes the extraction, the flat `.vd` accessor, grid-based
geometry, and flexible nonspatial dimension ordering. Downstream integration is
still pending.

## Status

### Done

- Standalone `uv` package `voxeldata` (`uv_build`, Python 3.11–3.14).
- `src/voxeldata/`:
  - `_dims.py` — `VOXEL_DIMS`, `WORLD_DIMS`, `GRID_DIM`, `TIME_DIM`, `CORE_DIMS`.
  - `_coordinates.py` — `get_representative_step`, `CoordinateSpacingInfo`,
    `get_coordinate_spacing_info`, `get_coordinate_origins` (new; returns first
    coordinate per dim, warns and returns `0.0` when a dim has no coordinate).
  - `geometry.py` — `VoxelToWorldIndex`/`VoxelToWorldTransform`, xarray index
    machinery, attach/get helpers, units/spacing/origin/direction helpers, and
    **named affine frames on the index** (`VoxelToWorldIndex.named_affines`).
  - `affine.py` — `apply_affine`, `get_relative_affine`, `reindex_voxels`,
    `reindex_voxels_like`, `get_bounding_box`.
  - `create.py` — generic `create_voxeldata` (no fUSI timing metadata), now accepts
    `affines=`.
  - `validation.py` — `validate_voxeldata`, `ensure_voxeldata`,
    `canonicalize_voxeldata`, `require_dataarray`, `require_positive_finite`.
  - `accessors.py` — flat `@xr.register_dataarray_accessor("vd")` exposing
    `VoxelDataAccessor`.
  - `__init__.py` — public API, imports `VoxelDataAccessor` for registration.
- Tooling/docs: `pyproject.toml`, `justfile`, `.pre-commit-config.yaml`,
  `.gitignore`, `LICENSE` (BSD-3-Clause), `NOTICE`, `README.md`, `zensical.toml`,
  `docs/index.md`, `docs/api/*.md` (geometry, create, validation, affine, dims,
  accessors), `docs/stylesheets/extra.css`. Docs use the ConfUSIus visual style
  (Space Grotesk + JetBrains Mono, custom light/dark palette, hero index, nav tabs,
  admonition colors, rich Markdown extensions).
- `.github/workflows/lint.yml` and `test.yml` run on main pushes, PRs, manual
  dispatch, and reusable calls. Tests cover Python 3.11–3.14 on Linux/macOS/Windows.
- `.github/workflows/publish.yml` publishes `v*` tags through PyPI Trusted
  Publishing after calling lint/tests and checking tag/version agreement, docs,
  and build. Requires the publisher setup described below.
- Renamed the geometry dimension and public API to `grid`; arbitrary leading
  dimension order is accepted, while canonicalization uses `grid, time`.
- Ported modality-independent constructor, geometry, validation, bounding-box,
  and reindexing regressions from ConfUSIus. Shared function-scoped fixtures live
  in `tests/conftest.py`; distinguishable samples and noncommuting transforms
  avoid vacuous selection and composition checks.
- Fixed `reindex_voxels_like` to retain the source's named frames and compare
  grid-dependent world coordinates, rejecting mismatched grid labels.
- Verified: `ruff`, `ty`, `numpydoc`, `codespell`, pre-commit hooks, `pytest`
  (154 passing on Python 3.11–3.14 on Linux; 90% statement coverage), `uv build`,
  `zensical build --strict`. Temporary fault injection confirmed tests catch reversed
  relative transforms and swapped world-coordinate outputs.

### Remaining

1. **Integrate into ConfUSIus** (not started):
   - Replace `confusius._utils.geometry` and `confusius.xarray.affine` internals
     with re-exports from `voxeldata`, or make `confusius` depend on `voxeldata`
     and delete duplicated code.
   - Remove all `attrs["affines"]` usage in ConfUSIus; move consumers to
     `data.vd.affines` / `get_affine` / `set_affine` / `apply` / `to`.
   - Keep ConfUSIus's `.fusi` accessor as the fUSI-specific layer, delegating
     geometry to `voxeldata`; do not duplicate geometry logic.
2. **Update voxreg** to consume `voxeldata` named affines (`data.vd.to(other, via)`)
   and drop `attrs["affines"]`.
3. Optional further test ports: advanced vectorized indexers and lazy-array
   backend behavior. The standalone suite does not require Dask or fUSI metadata.
4. Configure the GitHub `pypi` environment and PyPI Trusted Publisher before the
   first release. The workflows are present; no release has been triggered.
5. Optionally add a `CHANGELOG.md`.

## Design decisions (do not silently change)

- **Flat `.vd` root accessor**, not nested `.vd.affine`. No `.voxeldata` alias.
- Model: `(*nonspatial_dims, k, j, i)`, with arbitrary leading dimension order.
  `time` and `grid` are special nonspatial dimensions. Creation preserves leading
  order; validation only requires a trailing spatial block in `k, j, i` order.
- Canonicalization and `ensure` use `(...extra_dims, grid, time, k, j, i)`, preserving
  the relative order of extra dimensions and omitting absent `grid`/`time` axes.
  Scalar-selected spatial axes are restored; a scalar-selected grid is not.
- `grid` selects a spatial sampling grid, with one affine per grid shared across
  time and every other nonspatial dimension. Different grids may have identical
  affines. There are no legacy geometry-dimension API aliases.
- Per-grid time coordinates may have dimensions `(time, grid)` or `(grid, time)`;
  validation checks increasing timestamps along the named `time` axis.
- Mutators default to `inplace=False`; `inplace=True` updates the original
  DataArray's coords/attrs and returns it.
- **Named affines live on `VoxelToWorldIndex.named_affines`**, never in
  `attrs["affines"]`.
- `attrs["affines"]` is **ignored entirely** — no read fallback, no mirror writes.
  `apply` does not read or rewrite that attr.
- `set_voxel_to_world` drops existing named frames by default and warns; pass
  `named_affines=` explicitly or use `apply()` to preserve/re-express frames.
- `apply(name)` sets the applied frame to an exact `(4, 4)` identity and
  re-expresses every other frame as `arr @ inv(action)` (using `np.matmul`, which
  broadcasts `2D @ 3D`).
- `equals()` ignores named affines: index equality stays geometry-only.
- Named affine value rules: `(4, 4)` shared frame or `(ngrid, 4, 4)` when geometry
  is grid-dependent; finite; homogeneous last row `[0, 0, 0, 1]`. Do **not** require
  equal scale magnitudes across grids (unlike `voxel_to_world`).
- `isel`:
  - spatial dims: named frames unchanged;
  - scalar `grid`: reduce `(ngrid, 4, 4)` to `(4, 4)`;
  - grid slice / array / list: slice axis 0.
- `concat` along `grid`: every index must carry the same frame names; all-2D frames
  must be `allclose` (keep one copy); all-3D frames concatenate along axis 0; mixing
  2D and 3D raises.
- `set_units`, `reindex_voxels`, `reindex_voxels_like` preserve the input's named
  frames. `reindex_voxels_like` never copies the reference's named frames; inputs
  with grid-dependent geometry must carry identical grid labels in the same order.
- For ConfUSIus compatibility, `.vd.spacing` and `.vd.origin` cover **all**
  dimensions; `.vd.direction`, `.vd.bounding_box`, `.vd.voxel_to_world`, and
  `.vd.scalings` are spatial-only.
- `.vd.set_voxel_to_world` is the only operation valid on a plain DataArray (no
  existing index); all other read ops raise `ValueError`.

## Public `.vd` API

Properties: `index`, `voxel_dims`, `world_dims`, `is_grid_dependent`,
`is_axis_aligned`, `fixed_voxel_coords`, `units`, `voxel_to_world`, `origin`,
`spacing`, `scalings`, `direction`, `bounding_box`, `affines`.

Methods:

```python
set_voxel_to_world(voxel_to_world, *, units=None, named_affines=None, inplace=False)
set_units(units, *, inplace=False)
apply(affine_or_name, *, inplace=False)
reindex_voxels(*, inplace=False)
reindex_voxels_like(reference, *, atol=1e-6, inplace=False)
to(other, via)
get_affine(name)
set_affine(name, affine, *, inplace=False)
remove_affine(name, *, inplace=False)
validate(**kwargs)
canonicalize()
ensure(**kwargs)
```

## Conventions

Follow the user's global Python guidelines plus ConfUSIus conventions:

- `uv` for env/deps; `just` for tasks.
- Absolute imports only.
- NumPy docstrings on every function/method, including private ones.
  - Optional: `arg : type, optional` or `arg : type, default: value`.
  - Booleans start "Whether to ...".
  - Multiple returns: one `name : type` block each.
  - Inline code uses single backticks.
  - Reference private objects as plain backticks, not `[][]` links.
- `ruff` lint/format, `ty` type check, `numpydoc` validation, `codespell`.
- Comments end with a period; no comments that duplicate code.
- Tests: public API only, meaningful assertions, `pytest`, seeded RNG when needed,
  small arrays. Reuse global fixtures from `tests/conftest.py` for common geometry;
  keep inputs specific to a constructor/error case local. Fixtures are function
  scoped so mutation tests cannot leak state. Use distinct values for data-order
  assertions and noncommuting matrices for composition assertions.

## Publishing setup

- Create the GitHub environment `pypi`; restrict deployments to `v*` release tags
  and optionally require approval.
- Register a PyPI Trusted Publisher (a pending publisher for the first release):
  project `voxeldata`, owner `confusius-tools`, repository `voxeldata`, workflow
  filename `publish.yml`, environment `pypi`. No API token is needed.
- Set `project.version`, run `uv lock`, commit/push, then push a matching `vVERSION`
  tag. The initial version is `0.1.0`; its matching tag is `v0.1.0`.
- Publishing is permanent. Use a new version for subsequent releases.
- Keep publishing instructions out of the README.

## Commands

```bash
uv sync
uv run ruff check --fix src tests
uv run ruff format src tests
uv run ty check
uv run pytest -q
uv run numpydoc lint src/voxeldata/*.py
uv run codespell src tests docs README.md zensical.toml justfile pyproject.toml
uv run zensical build --strict
uv build

just test
just pre-commit
just docs
```

Run one test:

```bash
uv run pytest tests/test_accessors.py::test_apply_named_affine_makes_it_identity
```

## File map

- `src/voxeldata/geometry.py` — index + named-affine implementation:
  - `VoxelToWorldIndex.__init__`, `with_named_affines`
  - `_normalize_named_affines`, `_concat_named_affines`
  - `VoxelToWorldIndex.rename`, `.isel`, `._isel_named_affines`, `.concat`
  - `attach_voxel_to_world_index(..., named_affines=None)`
  - `get_voxel_to_world_index`
- `src/voxeldata/affine.py` — `apply_affine` and `get_relative_affine` use
  `VoxelToWorldIndex.named_affines`; `_require_voxel_to_world_index` helper.
- `src/voxeldata/create.py` — `create_voxeldata(..., affines=None)`.
- `src/voxeldata/accessors.py` — `VoxelDataAccessor` and `_maybe_inplace`.
- `src/voxeldata/_coordinates.py` — `get_coordinate_origins`.
- `tests/conftest.py` — shared plain, anisotropic, oblique, irregular, time-series,
  and grid-dependent volumes, plus noncommuting named frames.
- `tests/test_accessors.py` — accessor contracts, named-frame validation,
  composition, slicing, concatenation, and in-place isolation.
- `tests/test_voxeldata.py` — core smoke tests and flexible dimension ordering.
- `tests/test_create.py` — constructor padding, ordering, metadata, and errors.
- `tests/test_geometry.py` — world selection, grid geometry, and alignment.
- `tests/test_validation.py` — coordinate, spacing, time, dtype, and recovery checks.
- `tests/test_affine.py` — independently evaluated bounds, reindexing, and frames.
- `docs/api/accessors.md` + `zensical.toml` nav entry.

## Known gotchas

- `create_voxeldata` requires either `spacing` or `voxel_to_world`; a bare
  `dims=("k","j","i")` call raises.
- A `grid` dim always requires a per-grid `(ngrid, 4, 4)` affine stack, even for one
  grid.
- `VoxelToWorldIndex.named_affines` values are always `z/y/x` world-frame `4x4`
  matrices, independent of how many voxel dims are active.
- `_replace_index` reattaches through `attach_voxel_to_world_index`, which rebuilds
  world coordinates from the current data; it is used by `set_units`,
  `set_affine`, and `remove_affine`.
