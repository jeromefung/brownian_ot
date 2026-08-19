"""Tests for ASDF metadata written by non-OT simulations.

FreeDiffusionSimulation and ConstantForceSimulation do not need MATLAB or
MSTM, so they are used here as a quick check of the metadata tree.
"""

import asdf
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from brownian_ot.particles import Dimer, Sphere, Spheroid
from brownian_ot.simulation import ConstantForceSimulation, FreeDiffusionSimulation


ETA = 1e-3
KT = 1.38e-23 * 295
DT = 1e-5
N_STEPS = 5
SEED = 12345
POS0 = np.array([1e-7, -2e-7, 3e-7])
ORIENT0 = np.identity(3)
EXPECTED_COLUMNS = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def _assert_common_metadata(tree, sim_class, particle, n_steps, pos0, seed):
    assert tree["schema_version"] == "1.0"
    assert "beam" not in tree

    sim_meta = tree["simulation"]
    assert sim_meta["class"] == sim_class
    assert sim_meta["n_steps"] == n_steps
    assert sim_meta["timestep"] == DT
    assert sim_meta["viscosity"] == ETA
    assert sim_meta["kT"] == KT
    assert sim_meta["rng_seed"] == seed
    assert_allclose(sim_meta["pos0"], pos0)
    assert_allclose(sim_meta["orient0"], np.array([1.0, 0.0, 0.0, 0.0]))
    assert "c" not in sim_meta

    part_meta = tree["particle"]
    assert part_meta["type"] == type(particle).__name__
    assert_allclose(part_meta["Ddim"], particle.Ddim)
    assert_allclose(part_meta["cod"], particle.cod)
    if particle.n_p is None:
        assert part_meta["n_p"] is None
    else:
        assert_allclose(np.asarray(part_meta["n_p"]), np.asarray(particle.n_p))

    traj_meta = tree["trajectory"]
    assert traj_meta["n_rows"] == n_steps + 1
    assert traj_meta["columns"] == EXPECTED_COLUMNS

    data = np.asarray(tree["data"])
    assert data.shape == (n_steps + 1, 7)
    assert_allclose(data[0, :3], pos0)
    assert_allclose(data[0, 3:], np.array([1.0, 0.0, 0.0, 0.0]))


def test_free_diffusion_sphere_metadata():
    particle = Sphere(a=1e-6)
    sim = FreeDiffusionSimulation(
        particle,
        DT,
        ETA,
        KT,
        pos0=POS0,
        orient0=ORIENT0,
        seed=SEED,
    )
    output = sim.run(N_STEPS)

    assert isinstance(output, asdf.AsdfFile)
    _assert_common_metadata(
        output.tree, "FreeDiffusionSimulation", particle, N_STEPS, POS0, SEED
    )
    assert "force" not in output["simulation"]
    assert output["particle"]["a"] == particle.a
    assert "ar" not in output["particle"]
    assert "n_spheres" not in output["particle"]


def test_free_diffusion_spheroid_metadata():
    particle = Spheroid(a=2e-8, ar=5)
    sim = FreeDiffusionSimulation(
        particle, DT, ETA, KT, pos0=np.zeros(3), orient0=ORIENT0, seed=SEED
    )
    output = sim.run(N_STEPS)

    _assert_common_metadata(
        output.tree,
        "FreeDiffusionSimulation",
        particle,
        N_STEPS,
        np.zeros(3),
        SEED,
    )
    part_meta = output["particle"]
    assert part_meta["a"] == particle.a
    assert part_meta["ar"] == particle.ar
    assert part_meta["equivalent_sphere_radius"] == particle.equivalent_sphere_radius


def test_constant_force_metadata():
    particle = Sphere(a=1e-6)
    force = np.array([0.0, 0.0, -1e-12, 0.0, 0.0, 0.0])
    sim = ConstantForceSimulation(
        particle,
        DT,
        force,
        ETA,
        KT,
        pos0=POS0,
        orient0=ORIENT0,
        seed=SEED,
    )
    output = sim.run(N_STEPS)

    _assert_common_metadata(
        output.tree, "ConstantForceSimulation", particle, N_STEPS, POS0, SEED
    )
    assert_allclose(output["simulation"]["force"], force)
    assert "beam" not in output.tree


def test_dimer_particle_metadata():
    particle = Dimer(a=5e-7)
    sim = FreeDiffusionSimulation(
        particle, DT, ETA, KT, pos0=np.zeros(3), orient0=ORIENT0, seed=SEED
    )
    output = sim.run(N_STEPS)

    part_meta = output["particle"]
    assert part_meta["type"] == "Dimer"
    assert part_meta["n_spheres"] == particle.n_spheres
    assert_allclose(part_meta["sphere_pos"], particle.sphere_pos)
    assert part_meta["a"] == particle.a
    assert part_meta["equivalent_sphere_radius"] == particle.equivalent_sphere_radius
    assert "a_ratios" not in part_meta


def test_asdf_file_roundtrip(tmp_path):
    particle = Sphere(a=1e-6, n_p=1.45)
    force = np.array([1e-13, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim = ConstantForceSimulation(
        particle,
        DT,
        force,
        ETA,
        KT,
        pos0=POS0,
        orient0=ORIENT0,
        seed=SEED,
    )
    outfname = tmp_path / "const_force_meta"
    output = sim.run(N_STEPS, outfname=str(outfname))

    written = tmp_path / "const_force_meta.asdf"
    assert written.is_file()

    with asdf.open(written) as af:
        _assert_common_metadata(
            af.tree, "ConstantForceSimulation", particle, N_STEPS, POS0, SEED
        )
        assert_allclose(af["simulation"]["force"], force)
        assert_allclose(np.asarray(af["particle"]["n_p"]), np.asarray(particle.n_p))
        assert_array_equal(np.asarray(af["data"]), np.asarray(output["data"]))
