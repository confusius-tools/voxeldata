"""Internal comparisons of voxel-to-world geometry."""

import numpy as np
import numpy.typing as npt


def voxel_affines_match(
    first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]
) -> bool:
    """Compare voxel-to-world matrices in units of their voxel axes.

    Parameters
    ----------
    first, second : (..., 4, 4) numpy.ndarray
        Affines with broadcast-compatible leading geometry dimensions.

    Returns
    -------
    bool
        Whether the linear terms and homogeneous row agree and the origins
        differ by at most 1e-5 times the length of the shortest voxel axis.
    """
    first_basis = first[..., :3, :3]
    second_basis = second[..., :3, :3]
    if not np.allclose(first_basis, second_basis) or not np.allclose(
        first[..., 3, :], second[..., 3, :]
    ):
        return False
    shortest_axis = np.minimum(
        np.linalg.vector_norm(first_basis, axis=-2).min(axis=-1),
        np.linalg.vector_norm(second_basis, axis=-2).min(axis=-1),
    )
    # A common world translation must not change which physical grids align.
    origin_error = np.linalg.vector_norm(
        first[..., :3, 3] - second[..., :3, 3], axis=-1
    )
    return bool(np.all(origin_error <= 1e-5 * shortest_axis))
