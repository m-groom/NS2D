"""Tests for checkpoint discovery and auxiliary (forcing/spectra) state I/O."""

import numpy as np
from mpi4py import MPI

from ns2d import solver


def test_auxiliary_state_round_trip(tmp_path):
    """Auxiliary state written at a checkpoint reloads unchanged.

    `step_counter` is stored as a shape-(1,) array, so it must be read as an
    element rather than with a bare `int()`, which raises TypeError under
    NumPy >= 2 on rank 0 while every other rank blocks in the broadcast.
    """
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(0)
    forcing_state = {
        'state_potential': rng.standard_normal((4, 5)),
        'scale_state': np.array([1.0]),
        'step_counter': np.array([37], dtype=np.int64),
    }

    solver.save_auxiliary_state(
        tmp_path, 2.0, forcing_state, {'next_spec_t': 4.0}, comm
    )
    loaded = solver.load_auxiliary_state(tmp_path, 2.0, comm)

    assert loaded is not None
    assert loaded['forcing']['step_counter'] == 37
    assert np.array_equal(loaded['forcing']['state_potential'], forcing_state['state_potential'])
    assert loaded['forcing']['scale_state'][0] == 1.0
    assert loaded['spectra']['next_spec_t'] == 4.0


def test_latest_checkpoint_is_the_highest_set_number(tmp_path):
    """Checkpoint sets are ordered numerically, not lexicographically."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for n in range(1, 13):
        (checkpoint_dir / f"checkpoints_s{n}.h5").touch()

    assert solver.find_latest_checkpoint(tmp_path).name == "checkpoints_s12.h5"
