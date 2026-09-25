"""Unit tests for LG_Graphics helpers that own real math / conventions.

Skip thin wrappers over SciPy / PyVista and movie I/O / routing. Each
docstring starts with ``Target:`` naming the symbol in ``graphics.py``.

Quick map
---------
as_rotation              → OtSim scalar-first → SciPy scalar-last
_pose_matrix             → 4×4 homogeneous pose
xcm_nondim               → asymmetric-trimer CM formula
cluster lab positions    → pos0 + (a * a_pos) @ R.T
waist / lg_intensity     → beam envelope + |u|² shape
_scale_movie_arrays      → which arrays get length_scale
_particle_pad_length     → pad radius / spheroid longest axis
_lab_bounds              → traj min/max ± pad * a
arrow                    → absolute tip/shaft sizes → pv.Arrow fractions
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from brownian_ot.graphics import LG_Graphics


# ---------------------------------------------------------------------------
# Orientation / pose math
# ---------------------------------------------------------------------------

def test_as_rotation_scalar_first_convention():
    """Target: LG_Graphics.as_rotation

    OtSim stores (w, x, y, z); SciPy wants (x, y, z, w). Wrong reorder still
    returns a Rotation — so assert the *matrix*, not the type.
    """
    s = np.sqrt(2) / 2
    q_otsim = [s, 0.0, 0.0, s]          # 90° about +z, scalar-first
    q_scipy = [0.0, 0.0, s, s]          # same quat, scalar-last
    R_ours = LG_Graphics.as_rotation(q_otsim).as_matrix()
    R_ref = Rotation.from_quat(q_scipy).as_matrix()
    assert_allclose(R_ours, R_ref, atol=1e-12)
    assert_allclose(R_ours @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_pose_matrix_applies_R_then_translation():
    """Target: LG_Graphics._pose_matrix

    M @ [v; 1] == concatenate(R @ v + pos, [1]).
    """
    R = np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]])
    pos = np.array([1.0, 2.0, 3.0])
    v = np.array([0.5, -0.25, 1.0])
    M = LG_Graphics._pose_matrix(R, pos)
    out = M @ np.append(v, 1.0)
    expected = np.append(R @ v + pos, 1.0)
    assert_allclose(out, expected, atol=1e-12)


def test_pose_matrix_origin_maps_to_pos():
    """Target: LG_Graphics._pose_matrix

    Body origin [0,0,0,1] maps to [pos, 1].
    """
    R = np.eye(3)
    pos = np.array([-1.0, 4.0, 0.5])
    M = LG_Graphics._pose_matrix(R, pos)
    assert_allclose(M @ [0, 0, 0, 1], [pos[0], pos[1], pos[2], 1.0])


# ---------------------------------------------------------------------------
# Trimer / cluster geometry
# ---------------------------------------------------------------------------

def test_xcm_nondim_formula():
    """Target: LG_Graphics.xcm_nondim

    xcm_nondim(c) == c³ √(c²+2c) / (c³+2); c→0 → 0.
    """
    for c in (0.0, 0.25, 1.0, 2.0):
        expected = c**3 * np.sqrt(c**2 + 2 * c) / (c**3 + 2) if c > 0 else 0.0
        assert_allclose(LG_Graphics.xcm_nondim(c), expected, atol=1e-12)


def test_cluster_lab_positions_formula():
    """Target: LG_Graphics.add_sphere_cluster  (math only — no Plotter)

    lab = pos0 + (a * a_pos) @ R.T   ⇔   lab[i] = pos0 + R @ (a * a_pos[i])
    """
    a = 1.5
    pos0 = np.array([0.1, -0.2, 0.3])
    a_pos = np.array([[0.0, 0.0, 1.0],
                      [0.0, 0.0, -1.0],
                      [0.5, 0.0, 0.0]])
    s = np.sqrt(2) / 2
    R = LG_Graphics.as_rotation([s, 0.0, 0.0, s]).as_matrix()
    lab = pos0 + (a * a_pos) @ R.T
    for i, body in enumerate(a_pos):
        assert_allclose(lab[i], pos0 + R @ (a * body), atol=1e-12)
    # dimer ± bond along body z
    assert_allclose(lab[0], pos0 + R @ [0, 0, a], atol=1e-12)
    assert_allclose(lab[1], pos0 - R @ [0, 0, a], atol=1e-12)


# ---------------------------------------------------------------------------
# LG beam math
# ---------------------------------------------------------------------------

def test_waist_focus_and_rayleigh():
    """Target: LG_Graphics.waist

    w(0)=w0; w(±zR)=w0√2; even in z.
    """
    w0, zR = 0.6, 1.6
    assert_allclose(LG_Graphics.waist(0.0, w0, zR), w0)
    assert_allclose(LG_Graphics.waist(zR, w0, zR), w0 * np.sqrt(2))
    assert_allclose(
        LG_Graphics.waist(-zR, w0, zR), LG_Graphics.waist(zR, w0, zR)
    )


def test_lg_intensity_mode_shape():
    """Target: LG_Graphics.lg_intensity

    ℓ=0 peaks on axis; |ℓ|≥1 has a dark core; intensity depends on |ℓ| only.
    """
    kw = dict(p=0, w0=0.6, zR=1.6)
    assert LG_Graphics.lg_intensity(0, 0, 0, ell=0, **kw) > LG_Graphics.lg_intensity(
        0.5, 0, 0, ell=0, **kw
    )
    assert_allclose(LG_Graphics.lg_intensity(0, 0, 0, ell=2, **kw), 0.0, atol=1e-12)
    assert_allclose(
        LG_Graphics.lg_intensity(0.4, 0.1, 0.2, ell=2, **kw),
        LG_Graphics.lg_intensity(0.4, 0.1, 0.2, ell=-2, **kw),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Movie array scaling / bounds (policy math — not I/O)
# ---------------------------------------------------------------------------

def test_scale_movie_arrays_what_scales():
    """Target: LG_Graphics._scale_movie_arrays

    traj xyz, radius, perpendicular_radius, center_of_diffusion × length_scale.
    Quats, sphere_positions, aspect_ratios, aspect_ratio unchanged.
    """
    scale = 1e6
    traj = np.array([[1e-6, 2e-6, 3e-6, 1.0, 0.0, 0.0, 0.0],
                     [4e-6, 5e-6, 6e-6, 0.0, 1.0, 0.0, 0.0]])
    part = {
        'type': 'SphereCluster',
        'radius': 1e-6,
        'perpendicular_radius': 2e-6,
        'aspect_ratio': 1.5,
        'sphere_positions': np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        'aspect_ratios': np.array([1.0, 0.5]),
        'center_of_diffusion': np.array([1e-7, 0.0, 0.0]),
    }
    traj0 = traj.copy()
    sp0 = part['sphere_positions'].copy()
    ar0 = part['aspect_ratios'].copy()

    out_traj, out_part = LG_Graphics._scale_movie_arrays(traj, part, scale)

    assert_allclose(out_traj[:, :3], traj0[:, :3] * scale)
    assert_allclose(out_traj[:, 3:], traj0[:, 3:])
    assert_allclose(out_part['radius'], 1.0)
    assert_allclose(out_part['perpendicular_radius'], 2.0)
    assert out_part['aspect_ratio'] == 1.5
    assert_allclose(out_part['sphere_positions'], sp0)
    assert_allclose(out_part['aspect_ratios'], ar0)
    assert_allclose(out_part['center_of_diffusion'], [0.1, 0.0, 0.0])
    # inputs not mutated
    assert_allclose(traj, traj0)
    assert_allclose(part['sphere_positions'], sp0)


def test_scale_movie_arrays_skips_center_of_diffusion_none():
    """Target: LG_Graphics._scale_movie_arrays

    center_of_diffusion is None → stays None.
    """
    traj = np.zeros((1, 7))
    part = {'radius': 1.0, 'center_of_diffusion': None}
    _, out = LG_Graphics._scale_movie_arrays(traj, part, 1e6)
    assert out['center_of_diffusion'] is None


def test_particle_pad_length():
    """Target: LG_Graphics._particle_pad_length

    radius preferred; spheroid → a * max(1, |ar|); empty → 1.0.
    """
    assert LG_Graphics._particle_pad_length({'radius': 0.3}) == 0.3
    assert LG_Graphics._particle_pad_length(
        {'radius': 0.3, 'perpendicular_radius': 9.0}
    ) == 0.3
    assert LG_Graphics._particle_pad_length(
        {'perpendicular_radius': 0.2, 'aspect_ratio': 3.0}
    ) == pytest.approx(0.6)
    assert LG_Graphics._particle_pad_length(
        {'perpendicular_radius': 0.2, 'aspect_ratio': 0.5}
    ) == pytest.approx(0.2)
    assert LG_Graphics._particle_pad_length(
        {'perpendicular_radius': 0.2, 'aspect_ratio': -2.0}
    ) == pytest.approx(0.4)
    assert LG_Graphics._particle_pad_length({}) == 1.0


def test_lab_bounds_pads_xyz_only():
    """Target: LG_Graphics._lab_bounds

    Bounds from traj[:, :3] ± pad*a; quaternion columns ignored.
    """
    traj = np.array([
        [0.0, 1.0, -2.0, 99.0, 99.0, 99.0, 99.0],
        [2.0, 3.0,  4.0, -99.0, -99.0, -99.0, -99.0],
    ])
    a, pad = 0.5, 10.0
    # instance method that does not use self / Plotter
    bnds = LG_Graphics.__new__(LG_Graphics)._lab_bounds(traj, a, pad=pad)
    d = pad * a
    assert_allclose(bnds, [
        0.0 - d, 2.0 + d,
        1.0 - d, 3.0 + d,
        -2.0 - d, 4.0 + d,
    ])


# ---------------------------------------------------------------------------
# Arrow size conversion (our glue over pv.Arrow)
# ---------------------------------------------------------------------------

def test_arrow_absolute_sizes_to_fractions(monkeypatch):
    """Target: LG_Graphics.arrow

    tip_len/tip_r/shaft_r are absolute scene units → fractions of ``scale``.
    Direction is normalized.
    """
    captured = {}

    def fake_arrow(**kwargs):
        captured.update(kwargs)
        return 'mesh'

    monkeypatch.setattr('brownian_ot.graphics.pv.Arrow', fake_arrow)
    length = 2.0
    LG_Graphics.arrow(
        (0, 0, 0), (3, 0, 0), length, shaft_r=0.1, tip_r=0.2, tip_len=0.5
    )
    assert_allclose(captured['direction'], [1.0, 0.0, 0.0])
    assert captured['scale'] == length
    assert_allclose(captured['tip_length'], 0.5 / length)
    assert_allclose(captured['tip_radius'], 0.2 / length)
    assert_allclose(captured['shaft_radius'], 0.1 / length)
