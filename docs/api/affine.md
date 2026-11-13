# Affine operations

World transforms update geometry without resampling or computing signal data.
Use the flat accessor methods `.vd.translate()`, `.vd.rotate()`, `.vd.zoom()`,
and `.vd.flip()`, or the corresponding `apply_*` functions below. Named affine
frames are re-expressed relative to the transformed world frame.

The default `center="geometry"` uses the midpoint of the bounding box enclosing
**all grids**, providing one shared pivot as in ConfUSIus. `.vd.bounding_box`
returns a dictionary mapping `z`, `y`, `x` to `(min, max)` tuples;
`.vd.geometry_center` maps the same keys to floats, matching `.vd.origin`.
Values use `.vd.world_units`. These are geometry summaries rather than sampled
arrays; they do not carry dimensions or coordinate attributes.
Select geometry dependencies first for grid-specific bounds or centers, e.g.
`data.sel(pose=pose_label).vd.bounding_box` or
`data.sel(pose=pose_label).vd.geometry_center` when geometry depends on `pose`.

Bounds and centers previously returned DataArrays. Use `bounding_box["x"][0]`
instead of `bounding_box.sel(bound="min", component="x")`, and
`geometry_center["x"]` instead of `geometry_center.sel(component="x")`.
Retrieve units from `.vd.world_units`, not result attributes.

`center="origin"` uses the first sampled voxel, including after cropping or
reversing native labels, and requires selecting all geometry dependencies first.
A world `(z, y, x)` point or mapping with those keys also supplies a shared pivot.

Rotations follow the right-hand rule in ordered world `(z, y, x)` coordinates.
Multiple axis names apply sequentially about fixed world axes, from left to right.
Zoom acts along world axes, so anisotropic zoom may shear oblique grids. Flip
reflects geometry rather than reversing samples.

All operations default to `inplace=False`. Non-inplace transforms restore
scalar-selected voxel dimensions; `inplace=True` requires all native voxel dims
and returns the updated original. Irregular voxel coordinates cannot provide a
bounding-box center; use an explicit pivot or `center="origin"` instead.

::: voxeldata.affine
