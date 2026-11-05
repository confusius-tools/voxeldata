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
- Renamed the geometry dimension and public API to `grid`; arbitrary leading
  dimension order is accepted, while canonicalization uses `grid, time`.
- Verified: `ruff`, `ty`, `numpydoc`, `codespell`, `pytest` (34 passing),
  `uv build`, `zensical build --strict`.

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
3. **Tests to add** (current `tests/test_accessors.py` covers the main paths):
   - Named-affine validation errors: wrong shape, non-finite, bad last row, 3D
     frame without grid-dependent geometry.
   - `concat` error paths: different frame names, differing shared frames, mixed
     2D/3D frames.
   - `isel(grid=slice/list)` slicing named-affine stacks.
   - `inplace=True` behaviors.
   - `get_relative_affine` missing-frame `KeyError`.
4. Optionally add a `CHANGELOG.md` and CI workflow.

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
- `set_units`, `reindex_voxels`, `reindex_voxels_like` preserve named frames.
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
- Tests: public API only, meaningful assertions, `pytest`, seeded RNG, small arrays.

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
- `tests/test_accessors.py` — 12 accessor/named-affine tests.
- `tests/test_voxeldata.py` — 22 core geometry/creation/validation cases, including
  flexible leading order, canonicalization, per-grid timestamps, and affine stacks.
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
