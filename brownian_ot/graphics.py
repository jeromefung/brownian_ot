'''
PyVista helpers for visualizing brownian_ot particles and ASDF trajectories.

LG_Graphics is a thin wrapper around pyvista.Plotter. It knows:
- OtSim / brownian_ot quaternion order (w, qx, qy, qz)
- How ASDF particle metadata maps onto meshes (Sphere, Spheroid, Dimer,
  SphereCluster)
- How to turn a trajectory into a single-pane or side-by-side mp4

Scene units are whatever you choose for static demos; for movies from SI ASDF
files use length_scale=1e6 to work in micrometers.
'''

from pathlib import Path

import asdf
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from scipy.special import genlaguerre


class MovieParamError(Exception):
    '''Invalid combination of movie input arguments.'''


class UnknownParticle(Exception):
    '''ASDF particle type that LG_Graphics cannot draw.'''


class LG_Graphics:
    '''
    Plotter-backed drawing API for brownian_ot particles and LG-beam chrome.

    Typical use
    -----------
    Static scene::

        g = LG_Graphics()
        g.setRotationFromEuler([20, 35, 10])
        g.add_dimer(0.5, np.zeros(3), g.quat)
        g.show()

    Movie from ASDF (prefer add_lab_axes=False; movie replaces self.pl)::

        g = LG_Graphics(add_lab_axes=False)
        g.movie(asdf_file='traj.asdf', name='out', length_scale=1e6)
    '''

    def __init__(self, off_screen=False, add_lab_axes=True):
        '''
        Parameters
        ----------
        off_screen : bool
            Passed to ``pv.Plotter``. Useful on headless / WSL setups.
            ``movie`` / ``movie_lab`` always build their own plotter.
        add_lab_axes : bool
            If True, draw fixed lab x/y/z arrows on the initial plotter.
            Set False when you only need helpers or will call a movie method
            (those replace ``self.pl``).
        '''
        # Absolute arrow sizes used by add_axis_lab / _setup_lab_axes
        self.LabAxis_shaft_r = 0.03
        self.LabAxis_tip_r = 0.08
        self.LabAxis_tip_len = 0.4
        self.LabAxis_font_size = 34
        self.LabAxis_axisLength = 5

        self.pl = pv.Plotter(off_screen=off_screen)
        self.pl.camera_position = self._default_camera()
        if add_lab_axes:
            self.add_axis_lab()

    # ------------------------------------------------------------------
    # Orientation state (for interactive / static scenes)
    # ------------------------------------------------------------------

    def setRotationFromEuler(self, angles, seq='xyz', degrees=True):
        '''Store OtSim-order quat on ``self.quat`` from Euler angles.'''
        # SciPy returns (x, y, z, w); brownian_ot / OtSim use (w, x, y, z).
        _xyzw = Rotation.from_euler(seq, angles, degrees=degrees).as_quat()
        self.quat = np.array([_xyzw[3], _xyzw[0], _xyzw[1], _xyzw[2]])

    def setRotationFromQuant(self, quat):
        '''Store OtSim-order quat ``(w, qx, qy, qz)`` on ``self.quat``.'''
        self.quat = np.array([quat[0], quat[1], quat[2], quat[3]])

    def show(self):
        '''Display the current plotter window / notebook backend.'''
        self.pl.show()

    # ------------------------------------------------------------------
    # Axes / labels
    # ------------------------------------------------------------------

    @staticmethod
    def arrow(start, direction, length, shaft_r=0.05, tip_r=0.15, tip_len=0.5):
        '''
        Arrow mesh with *absolute* tip/shaft sizes (paper-figure friendly).

        ``pv.Arrow`` treats tip/shaft sizes as fractions of ``scale``. We convert
        absolute lengths so callers can think in scene units the way the older
        POV-Ray figures did.
        '''
        start = np.asarray(start, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        return pv.Arrow(
            start=start,
            direction=direction,
            scale=length,
            tip_length=tip_len / length,
            tip_radius=tip_r / length,
            shaft_radius=shaft_r / length,
            tip_resolution=48,
            shaft_resolution=48,
        )

    def add_axis(self, start, direction, length, label, text_color='black', **kw):
        '''
        One labelled axis: arrow mesh plus billboard text past the tip.

        Keyword overrides: ``color``, ``font_size``, ``label_offset``, plus any
        absolute sizes forwarded to ``arrow`` (``shaft_r``, ``tip_r``, …).
        '''
        color = kw.pop('color', 'black')
        font_size = kw.pop('font_size', 30)
        label_offset = kw.pop('label_offset', 0.55)

        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        self.pl.add_mesh(self.arrow(start, direction, length, **kw), color=color)

        tip = np.asarray(start, dtype=float) + direction * (length + label_offset)
        self.pl.add_point_labels(
            [tip],
            [label],
            font_size=font_size,
            text_color=text_color,
            italic=kw.get('italic', True),
            bold=kw.get('bold', False),
            shape=None,
            show_points=False,
            always_visible=True,
            font_file=str(
                Path(__file__).resolve().parent / 'fonts' / 'DejaVuSans.ttf'
            ),
        )

    def add_axis_lab(self, **kwargs):
        '''Draw fixed lab-frame x, y, z arrows from the origin.'''
        for direction, label in (((1, 0, 0), 'x'), ((0, 1, 0), 'y'), ((0, 0, 1), 'z')):
            self.add_axis(
                (0, 0, 0),
                direction,
                kwargs.get('axisLength', self.LabAxis_axisLength),
                label,
                shaft_r=kwargs.get('shaft_r', self.LabAxis_shaft_r),
                tip_r=kwargs.get('tip_r', self.LabAxis_tip_r),
                tip_len=kwargs.get('tip_len', self.LabAxis_tip_len),
                font_size=kwargs.get('font_size', self.LabAxis_font_size),
            )

    # ------------------------------------------------------------------
    # Quaternions ↔ lab / body frames (OtSim conventions)
    # ------------------------------------------------------------------

    @staticmethod
    def as_rotation(quat):
        '''
        OtSim-style quaternion → ``scipy.spatial.transform.Rotation``.

        Trajectory rows are ``x, y, z, w, qx, qy, qz`` with scalar-**first**
        quaternions (numpy-quaternion / OtSim). SciPy wants scalar-**last**,
        so we rearrange here.

        A full quaternion carries roll around the dimer bond; a single axis
        vector would not place an asymmetric trimer's small sphere uniquely.
        '''
        q = np.asarray(quat, dtype=float).reshape(4)
        # OtSim (w, x, y, z) → SciPy (x, y, z, w)
        return Rotation.from_quat([q[1], q[2], q[3], q[0]])

    @staticmethod
    def rotation_matrix(quat):
        '''
        3×3 matrix R with ``v_lab = R @ v_body``.

        Columns are body axes in lab coordinates: ``R = [u1 | u2 | u3]``.
        '''
        return LG_Graphics.as_rotation(quat).as_matrix()

    @staticmethod
    def particle_frame(quat):
        '''
        Body triad ``(u1, u2, u3)`` in the lab frame.

        Project convention (dimer / asymmetric trimer):
        - u3 — dimer bond (body +z)
        - u1 — small-sphere offset (body +x)
        - u2 — completes a right-handed frame (body +y)
        '''
        R = LG_Graphics.rotation_matrix(quat)
        u1 = R[:, 0]
        u2 = R[:, 1]
        u3 = R[:, 2]
        return u1, u2, u3

    def add_particle_axes(
        self,
        position,
        quat,
        length=2.6,
        shaft_r=0.04,
        tip_r=0.10,
        tip_len=0.4,
        font_size=30,
        label_offset=0.35,
    ):
        '''Draw body axes u1, u2, u3 at ``position`` (usually the COM).'''
        position = np.asarray(position, dtype=float)
        u1, u2, u3 = self.particle_frame(quat)
        for vec, label, color in (
            (u1, 'u₁', '#000000'),
            (u2, 'u₂', '#000000'),
            (u3, 'u₃', '#000000'),
        ):
            self.add_axis(
                position,
                vec,
                length,
                label,
                shaft_r=shaft_r,
                tip_r=tip_r,
                tip_len=tip_len,
                font_size=font_size,
                label_offset=label_offset,
                color=color,
            )

    @staticmethod
    def _pose_matrix(R, pos):
        '''4×4 rigid transform: body point ``v`` → ``R @ v + pos`` in lab.'''
        M = np.eye(4, dtype=float)
        M[:3, :3] = np.asarray(R, dtype=float)
        M[:3, 3] = np.asarray(pos, dtype=float)
        return M

    # ------------------------------------------------------------------
    # Particle meshes (named actors; pose via user_matrix for movies)
    # ------------------------------------------------------------------

    def add_sphere(self, a, pos, quat, texture=None, key=''):
        '''
        Add or update one sphere at ``pos``, oriented by ``quat``.

        Mesh is built once at the origin (UV map only if ``texture`` is set).
        Pose is applied with the actor ``user_matrix`` so later frames update
        the transform instead of remeshing.
        '''
        pos = np.asarray(pos, dtype=float)
        R = LG_Graphics.rotation_matrix(quat)
        name = key if key != '' else 'sphere-' + str(len(list(self.pl.actors)))
        M = self._pose_matrix(R, pos)

        if name in self.pl.actors:
            self.pl.actors[name].user_matrix = M
            return

        mesh = pv.Sphere(radius=a, center=(0, 0, 0))
        if texture:
            mesh = mesh.texture_map_to_sphere(prevent_seam=True)

        if texture:
            actor = self.pl.add_mesh(mesh, texture=texture, name=name)
        else:
            actor = self.pl.add_mesh(mesh, name=name)
        actor.user_matrix = M

    def add_spheroid(self, a, ar, pos, quat, texture=None, key=''):
        '''
        Add or update a spheroid (ellipsoid of revolution) at ``pos``.

        Matches brownian_ot ``Spheroid``: body +z is the symmetry axis,
        equatorial radius ``a`` (``perpendicular_radius``), polar radius
        ``a * ar`` (``aspect_ratio``). ``ar > 1`` prolate, ``ar < 1`` oblate.

        Same pose path as ``add_sphere`` (unit sphere scaled in body frame,
        then ``user_matrix``).
        '''
        pos = np.asarray(pos, dtype=float)
        R = LG_Graphics.rotation_matrix(quat)
        name = key if key != '' else 'spheroid-' + str(len(list(self.pl.actors)))
        M = self._pose_matrix(R, pos)

        if name in self.pl.actors:
            self.pl.actors[name].user_matrix = M
            return

        mesh = pv.Sphere(radius=1.0, center=(0, 0, 0))
        if texture:
            mesh = mesh.texture_map_to_sphere(prevent_seam=True)
        pts = np.asarray(mesh.points)
        mesh.points = pts * np.array([float(a), float(a), float(a) * float(ar)])

        if texture:
            actor = self.pl.add_mesh(mesh, texture=texture, name=name)
        else:
            actor = self.pl.add_mesh(mesh, name=name)
        actor.user_matrix = M

    def add_dimer(self, a, pos, quat, texture=None, key=''):
        '''
        Two equal spheres along body ±z, centre of mass at ``pos``.

        Body centres (touching when separation = 2a)::

            upper: (0, 0, +a)
            lower: (0, 0, -a)

        Lab positions: ``p = pos + R @ body``. Same layout as OtSim dimer
        animations and brownian_ot ``Dimer.sphere_pos``.
        '''
        pos = np.asarray(pos, dtype=float)
        R = LG_Graphics.rotation_matrix(quat)

        body_upper = np.array([0.0, 0.0, +a])
        body_lower = np.array([0.0, 0.0, -a])
        p_upper = pos + R @ body_upper
        p_lower = pos + R @ body_lower

        key = key if key != '' else 'dimer-' + str(len(list(self.pl.actors)))
        self.add_sphere(a, p_upper, quat, texture, key=key + '-1')
        self.add_sphere(a, p_lower, quat, texture, key=key + '-2')

    @staticmethod
    def xcm_nondim(c):
        '''
        Nondimensional CM shift for an asymmetric trimer.

        ``c = a_small / a_big``. Sphere centres are defined in the body frame
        with the CM at the origin; the two large spheres shift by ``-xcm`` along
        body +x. Formula matches brownian_ot / jf121721.
        '''
        return c**3 * np.sqrt(c**2 + 2 * c) / (c**3 + 2)

    def add_sphere_cluster(self, a, pos0, a_pos, quat, a_ratio, texture=None, key=''):
        '''
        Draw a ``SphereCluster`` (or dimer) from body-frame geometry.

        Parameters
        ----------
        a : float
            Dimensional radius scale (ASDF ``radius``).
        pos0 : array-like (3)
            Lab-frame COM for this frame.
        a_pos : array-like (n, 3)
            Body centres in units of ``a`` (ASDF ``sphere_positions``).
        quat : array-like (4)
            Orientation ``(w, qx, qy, qz)``.
        a_ratio : array-like (n)
            Per-sphere radius ratios (ASDF ``aspect_ratios``, or all ones).

        Lab centres: ``lab = pos0 + (a * a_pos) @ R.T`` (each row ``R @ body_i``).
        '''
        R = LG_Graphics.rotation_matrix(quat)
        lab = pos0 + (a * a_pos) @ R.T

        key = key if key != '' else 'dimer-' + str(len(list(self.pl.actors)))
        for i in range(len(a_pos)):
            self.add_sphere(
                a * a_ratio[i], lab[i], quat, texture, key=key + '-' + str(i)
            )

    # ------------------------------------------------------------------
    # Laguerre–Gaussian beam chrome (decorative; not the OTT beam object)
    # ------------------------------------------------------------------

    @staticmethod
    def waist(z, w0=0.6, zR=1.6):
        '''
        Gaussian beam radius vs propagation distance::

            w(z) = w0 * sqrt(1 + (z / zR)^2)
        '''
        return w0 * np.sqrt(1.0 + (np.asarray(z, dtype=float) / zR) ** 2)

    @staticmethod
    def gaussian_beam_surface(w0=0.6, zR=1.6, zmin=-4.5, zmax=4.5, nz=240, ntheta=160):
        '''
        Surface of revolution ``r = w(z)`` — outer beam envelope (not the doughnut).
        '''
        z = np.linspace(zmin, zmax, nz)
        theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
        Z, T = np.meshgrid(z, theta, indexing='ij')
        R = LG_Graphics.waist(Z, w0, zR)
        return pv.StructuredGrid(R * np.cos(T), R * np.sin(T), Z)

    @staticmethod
    def lg_intensity(x, y, z, ell=1, p=0, w0=0.6, zR=1.6):
        '''
        |u_{p,ℓ}|² shape for a Laguerre–Gaussian mode (overall scale free)::

            ρ = 2 r² / w(z)²
            |u|² ∝ (1/w)² · ρ^|ℓ| · [L_p^|ℓ|(ρ)]² · exp(−ρ)

        ``ell`` is the azimuthal / OAM index; ``p`` is the radial index.
        '''
        r = np.hypot(x, y)
        wz = LG_Graphics.waist(z, w0, zR)
        rho = 2.0 * r**2 / wz**2
        L = genlaguerre(p, abs(ell))(rho)
        amp = (1.0 / wz) * np.sqrt(rho) ** abs(ell) * L * np.exp(-0.5 * rho)
        return amp**2

    @staticmethod
    def lg_volume(
        ell=1, p=0, w0=0.6, zR=1.6, extent=2.6, zmin=-4.5, zmax=4.5, n_xy=110, n_z=150
    ):
        '''
        Sample normalized LG intensity onto ``pv.ImageData`` (``intensity``).

        Contour later with ``grid.contour(isosurfaces=[0.10], scalars='intensity')``.
        '''
        x = np.linspace(-extent, extent, n_xy)
        y = np.linspace(-extent, extent, n_xy)
        z = np.linspace(zmin, zmax, n_z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        I = LG_Graphics.lg_intensity(X, Y, Z, ell=ell, p=p, w0=w0, zR=zR)
        I /= I.max()  # isosurface 0.10 ≡ 10% of peak

        grid = pv.ImageData(
            dimensions=(n_xy, n_xy, n_z),
            spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]),
            origin=(x[0], y[0], z[0]),
        )
        grid.point_data['intensity'] = I.flatten(order='F')
        return grid

    def add_lg_beam(
        self,
        w0=0.6,
        zR=1.6,
        p=0,
        ell=1,
        zmin=-4.5,
        zmax=4.5,
        extent=2.6,
        isosurface=0.10,
        show_envelope=True,
        doughnut_color=(1.0, 0.35, 0.35),
        envelope_color=(0.35, 0.55, 1.0),
        doughnut_opacity=0.55,
        envelope_opacity=0.2,
    ):
        '''
        Draw an LG doughnut isosurface (+ optional Gaussian ``w(z)`` envelope).

        Parameters
        ----------
        w0, zR : float
            Waist and Rayleigh range (beam size / spreading).
        p, ell : int
            LG mode indices.
        zmin, zmax : float
            Drawn length along z.
        extent : float
            Half-width of the xy sampling box (raise for large |ell|).
        isosurface : float
            Contour level as a fraction of peak intensity.
        '''
        grid = self.lg_volume(
            ell=ell, p=p, w0=w0, zR=zR, extent=extent, zmin=zmin, zmax=zmax
        )
        self.pl.add_mesh(
            grid.contour(isosurfaces=[isosurface], scalars='intensity'),
            color=doughnut_color,
            opacity=doughnut_opacity,
            smooth_shading=True,
            show_scalar_bar=False,
            name='beam',
        )
        if show_envelope:
            self.pl.add_mesh(
                LG_Graphics.gaussian_beam_surface(w0, zR, zmin, zmax),
                color=envelope_color,
                opacity=envelope_opacity,
                show_scalar_bar=False,
                name='envelope',
            )

    # ------------------------------------------------------------------
    # Movie helpers (mirrors particle._prepare_metadata ownership)
    # ------------------------------------------------------------------

    @staticmethod
    def _materialize_particle(part):
        '''Copy ASDF particle mapping into a plain dict with ndarray values.'''
        out = dict(part)
        for key in (
            'radius',
            'sphere_positions',
            'aspect_ratios',
            'center_of_diffusion',
            'diffusion_tensor',
            'perpendicular_radius',
            'aspect_ratio',
        ):
            if key in out and out[key] is not None:
                out[key] = np.asarray(out[key]).copy()
        return out

    def _load_movie_data(self, asdf_trajectory, asdf_file):
        '''
        Return an in-memory ``{trajectory, particle}`` tree.

        Pass either an in-memory dict or an ASDF path (not both). File handles
        are closed after copying arrays so movies do not keep the file mapped.
        '''
        if asdf_trajectory is None:
            asdf_trajectory = {}
        has_tree = len(asdf_trajectory) > 0
        has_file = asdf_file != ''
        if has_tree and has_file:
            raise MovieParamError(
                "Can't have asdf_trajectory and asdf_file at the same time"
            )
        if has_tree:
            return {
                'trajectory': np.asarray(asdf_trajectory['trajectory']).copy(),
                'particle': self._materialize_particle(asdf_trajectory['particle']),
            }
        if has_file:
            with asdf.open(asdf_file) as af:
                return {
                    'trajectory': np.asarray(af['trajectory']).copy(),
                    'particle': self._materialize_particle(af['particle']),
                }
        raise MovieParamError('Need asdf_trajectory or asdf_file')

    def _draw_particle(self, part, pos, quat, key, texture=None):
        '''
        Dispatch to the mesh drawer that matches ASDF ``particle`` metadata.

        - ``Sphere`` → ``radius``
        - ``Spheroid`` → ``perpendicular_radius`` + ``aspect_ratio``
        - ``Dimer`` / ``SphereCluster`` with ``sphere_positions`` → cluster drawer
          (``Dimer`` subclasses ``SphereCluster`` in brownian_ot)
        - ``Dimer`` without positions → hardcoded ``add_dimer`` fallback
        '''
        ptype = part['type']

        if ptype == 'Sphere':
            self.add_sphere(part['radius'], pos, quat, texture=texture, key=key)
            return

        if ptype == 'Spheroid':
            self.add_spheroid(
                part['perpendicular_radius'],
                part['aspect_ratio'],
                pos,
                quat,
                texture=texture,
                key=key,
            )
            return

        # Prefer metadata geometry whenever sphere_positions is present.
        if 'sphere_positions' in part:
            a_pos = np.asarray(part['sphere_positions'])
            ratios = np.asarray(
                part['aspect_ratios'] if 'aspect_ratios' in part
                else np.ones(len(a_pos))
            )
            self.add_sphere_cluster(
                part['radius'], pos, a_pos, quat, ratios, texture=texture, key=key
            )
            return

        if ptype == 'Dimer':
            self.add_dimer(part['radius'], pos, quat, texture=texture, key=key)
            return

        raise UnknownParticle(f'Unsupported particle type: {ptype!r}')

    @staticmethod
    def _particle_pad_length(part):
        '''Length used to pad ``_lab_bounds`` (radius or longest spheroid semi-axis).'''
        if 'radius' in part:
            return float(part['radius'])
        if 'perpendicular_radius' in part:
            a = float(part['perpendicular_radius'])
            ar = float(part['aspect_ratio']) if 'aspect_ratio' in part else 1.0
            return a * max(1.0, abs(ar))
        return 1.0

    def _default_camera(self):
        '''Default ``[eye, look_at, up]`` for orientation / single-pane views.'''
        return [
            (13, 13, 8),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 1.0),
        ]

    def _setup_lab_axes(self):
        '''
        Fixed lab x/y/z arrows at the origin of the *current* subplot.

        Used by the right pane of ``movie_lab``. These axes do not rotate with
        the particle — only the particle quaternion changes each frame.
        '''
        for direction, label in (((1, 0, 0), 'x'), ((0, 1, 0), 'y'), ((0, 0, 1), 'z')):
            self.add_axis(
                (0, 0, 0),
                direction,
                self.LabAxis_axisLength,
                label,
                shaft_r=self.LabAxis_shaft_r,
                tip_r=self.LabAxis_tip_r,
                tip_len=self.LabAxis_tip_len,
                font_size=self.LabAxis_font_size,
            )

    def _lab_bounds(self, traj, a, pad=10.0):
        '''
        Axis-aligned box around the trajectory for the left pane of ``movie_lab``.

        Returns ``[xmin, xmax, ymin, ymax, zmin, zmax]`` with padding ``pad * a``
        so the particle stays inside the drawn wireframe.
        '''
        lo = traj[:, :3].min(axis=0) - pad * a
        hi = traj[:, :3].max(axis=0) + pad * a
        return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    def _add_bounds_axis_labels(self, bnds, pad_frac=0.05, font_size=28):
        '''
        One lab-frame x/y/z label on the +x (screen-right) side of the box.

        Assumes the left-pane camera looks roughly through the −y face with +z
        up, so +x is to the right of the frame (OtSim-style edge labels).
        '''
        xmin, xmax, ymin, ymax, zmin, zmax = bnds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        pad = pad_frac * max(xmax - xmin, ymax - ymin, zmax - zmin)
        # x: front bottom; y: right bottom depth; z: right front vertical
        points = [
            [cx, ymin, zmin - pad],
            [xmax - pad, cy, zmin],
            [xmax - pad, ymin, cz],
        ]
        self.pl.add_point_labels(
            points,
            ['x', 'y', 'z'],
            font_size=font_size,
            text_color='black',
            italic=True,
            shape=None,
            show_points=False,
            always_visible=True,
            font_file=str(
                Path(__file__).resolve().parent / 'fonts' / 'DejaVuSans.ttf'
            ),
        )

    def _write_frames(self, traj, part, draw_frame, slowdown=1, n_frames=None):
        '''
        Stride through ``traj`` and call ``draw_frame(pos, quat)`` each step.

        ``slowdown`` is a stride (keep every N-th row). ``n_frames`` caps how
        many mp4 frames are *written* after striding (not how many rows are kept).
        '''
        step = max(1, int(slowdown))
        written = 0
        limit = None if n_frames is None else int(n_frames)
        for i in range(0, traj.shape[0], step):
            if limit is not None and written >= limit:
                break
            draw_frame(traj[i, :3], traj[i, 3:])
            self.pl.write_frame()
            written += 1

    @staticmethod
    def _scale_movie_arrays(traj, part, length_scale):
        '''
        Scale SI trajectory / radii into scene units (e.g. 1e6 → micrometers).

        ``sphere_positions`` and ``aspect_ratios`` stay nondimensional.
        '''
        traj = np.asarray(traj, dtype=float).copy()
        traj[:, :3] *= length_scale
        part = dict(part)
        if 'radius' in part:
            part['radius'] = float(part['radius']) * length_scale
        if 'perpendicular_radius' in part:
            part['perpendicular_radius'] = (
                float(part['perpendicular_radius']) * length_scale
            )
        if 'center_of_diffusion' in part and part['center_of_diffusion'] is not None:
            part['center_of_diffusion'] = (
                np.asarray(part['center_of_diffusion'], dtype=float) * length_scale
            )
        return traj, part

    # ------------------------------------------------------------------
    # Public movie APIs
    # ------------------------------------------------------------------

    def movie(
        self,
        asdf_trajectory=None,
        asdf_file='',
        slowdown=1,
        name='output',
        framerate=24,
        quality=8,
        texture=None,
        off_screen=True,
        length_scale=1.0,
        n_frames=None,
    ):
        '''
        Single-pane lab movie: particle at full (pos, quat) each frame.

        Parameters
        ----------
        length_scale : float
            Multiply positions and radii (use ``1e6`` for m → µm).
        n_frames : int or None
            Max number of **written** mp4 frames after ``slowdown`` striding.
        '''
        if asdf_trajectory is None:
            asdf_trajectory = {}
        data = self._load_movie_data(asdf_trajectory, asdf_file)
        traj, part = self._scale_movie_arrays(
            data['trajectory'], data['particle'], length_scale
        )

        self.pl = pv.Plotter(off_screen=off_screen)
        self.pl.camera_position = self._default_camera()
        self._setup_lab_axes()
        self.pl.open_movie(name + '.mp4', framerate=framerate, quality=quality)

        def draw_frame(pos, quat):
            self._draw_particle(part, pos, quat, key='lab', texture=texture)

        self._write_frames(
            traj, part, draw_frame, slowdown=slowdown, n_frames=n_frames
        )
        self.pl.close()

    def movie_lab(
        self,
        asdf_trajectory=None,
        asdf_file='',
        slowdown=1,
        name='output',
        framerate=24,
        quality=8,
        texture=None,
        bounds=None,
        window_size=(1600, 800),
        off_screen=True,
        length_scale=1.0,
        n_frames=None,
    ):
        '''
        Two-pane side-by-side movie: lab motion | orientation at origin.

        Layout (PyVista ``shape=(1, 2)``)
        ---------------------------------
        Left ``(0, 0)`` — lab / trajectory view
            Particle at true COM with ``quat``. Wireframe box from ``bounds``
            or ``_lab_bounds``; one x/y/z label set via ``_add_bounds_axis_labels``.
            Camera looks through the −y face.

        Right ``(0, 1)`` — orientation-only view
            Same ``quat``, COM pinned at the origin. Fixed lab arrows from
            ``_setup_lab_axes``.

        Parameters
        ----------
        length_scale : float
            Multiply positions and radii (e.g. ``1e6`` for m → µm).
        n_frames : int or None
            Max written mp4 frames after ``slowdown`` (same as ``movie``).
            Bounds still use the full scaled trajectory unless ``bounds`` is set.
        bounds : sequence of 6 floats or None
            Optional ``[xmin, xmax, ymin, ymax, zmin, zmax]`` for the left box.
        window_size : tuple
            Pixel size of the combined dual-pane window.
        '''
        if asdf_trajectory is None:
            asdf_trajectory = {}

        data = self._load_movie_data(asdf_trajectory, asdf_file)
        traj, part = self._scale_movie_arrays(
            data['trajectory'], data['particle'], length_scale
        )
        a = self._particle_pad_length(part)
        cam = self._default_camera()

        self.pl = pv.Plotter(
            shape=(1, 2), window_size=window_size, off_screen=off_screen
        )

        # Left pane: lab motion inside a bounds box
        self.pl.subplot(0, 0)
        bnds = bounds if bounds is not None else self._lab_bounds(traj, a)
        # Titles blank — numeric ticks off; labels placed manually below.
        self.pl.show_bounds(
            bounds=bnds,
            location='outer',
            all_edges=True,
            xtitle=' ',
            ytitle=' ',
            ztitle=' ',
            color='black',
            show_xlabels=False,
            show_ylabels=False,
            show_zlabels=False,
        )

        xmin, xmax, ymin, ymax, zmin, zmax = bnds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        # Face-on through −y (+z up → +x is screen-right).
        dist = 2.4 * max(dx, dz)
        self.pl.camera_position = [
            (cx, ymin - dist, cz),
            (cx, cy, cz),
            (0.0, 0.0, 1.0),
        ]
        self._add_bounds_axis_labels(bnds)

        # Right pane: orientation at origin with fixed lab arrows
        self.pl.subplot(0, 1)
        self._setup_lab_axes()
        self.pl.camera_position = cam

        self.pl.open_movie(name + '.mp4', framerate=framerate, quality=quality)

        def draw_frame(pos, quat):
            self.pl.subplot(0, 0)
            self._draw_particle(part, pos, quat, key='lab', texture=texture)
            self.pl.subplot(0, 1)
            self._draw_particle(
                part, np.zeros(3), quat, key='ori', texture=texture
            )

        self._write_frames(
            traj, part, draw_frame, slowdown=slowdown, n_frames=n_frames
        )
        self.pl.close()
