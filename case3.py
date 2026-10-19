import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter


class Cell:
    def __init__(self, cell_id, initial_center, parameters):
        self.id = cell_id
        self.parameters = parameters
        self.is_emitting_signal = False
        self.signal_start_time = 0.0
        self.rho1 = parameters.get('CYTOPLASM_DENSITY', 1.2)
        self.rho2 = parameters.get('NUCLEUS_DENSITY', 1.4)

        n_nodes = parameters['N_NODES']
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        ring = np.column_stack((np.cos(angles), np.sin(angles)))
        center = np.asarray(initial_center, dtype=float)
        self.m = center + parameters['MEMBRANE_RADIUS'] * ring
        self.n = center + parameters['NUCLEUS_RADIUS'] * ring

        self.original_x_vectors = self.n - self.m
        self.original_y_vectors = self.m - np.roll(self.m, 1, axis=0)

        self.previous_m = self.m.copy()
        self.previous_n = self.n.copy()
        self.bonded = np.zeros(n_nodes, dtype=bool)
        self.restart = np.zeros(n_nodes, dtype=bool)

        self.center_of_mass = self.calculate_center_of_mass()
        self.original_hat_x_vectors = self.n - self.center_of_mass
        self.original_hat_y_vectors = self.n - np.roll(self.n, 1, axis=0)
        self.reach = 0.0

    def get_membrane_positions(self):
        return self.m

    def get_nucleus_positions(self):
        return self.n

    @staticmethod
    def _polygon_moments(points, weight):
        x, y = points[:, 0], points[:, 1]
        x_next, y_next = np.roll(x, -1), np.roll(y, -1)
        dx = x - x_next
        mx = weight / 6 * np.sum(dx * ((y + 2 * y_next) * x_next + (2 * y + y_next) * x))
        my = weight / 6 * np.sum(dx * (y ** 2 + y * y_next + y_next ** 2))
        area = weight / 2 * np.sum((y + y_next) * dx)
        return mx, my, area

    def calculate_center_of_mass(self):
        mx1, my1, a1 = self._polygon_moments(self.m, self.rho1)
        mx2, my2, a2 = self._polygon_moments(self.n, self.rho2 - self.rho1)
        area = a1 + a2
        if abs(area) < 1e-10:
            return self.m.mean(axis=0)
        return np.array([(mx1 + mx2) / area, (my1 + my2) / area])


class ChemicalSignalModule:
    def __init__(self, parameters):
        self.sources = []
        self.params = parameters

    def add_source(self, cell):
        self.sources.append(cell)

    def calculate_gradient(self, positions, t, exclude_id=None):
        total = np.zeros_like(positions)
        mu, amp = self.params['MU'], self.params['A']
        for source in self.sources:
            if not source.is_emitting_signal or t < source.signal_start_time or source.id == exclude_id:
                continue
            effective_time = t - source.signal_start_time + self.params['T_EPSILON']
            diff = positions - source.center_of_mass
            dist_sq = np.sum(diff ** 2, axis=1)
            coefficient = -amp / (2 * mu ** 2 * effective_time ** 2)
            total += coefficient * diff * np.exp(-dist_sq / (4 * mu * effective_time))[:, None]
        return total

    def calculate_concentration_field(self, X, Y, t):
        total = np.zeros_like(X)
        for source in self.sources:
            if not source.is_emitting_signal or t < source.signal_start_time:
                continue
            effective_time = t - source.signal_start_time + self.params['T_EPSILON']
            dist_sq = (X - source.center_of_mass[0]) ** 2 + (Y - source.center_of_mass[1]) ** 2
            denominator = self.params['MU'] * effective_time
            total += np.nan_to_num((self.params['A'] / denominator) *
                                   np.exp(-dist_sq / (4 * denominator)))
        return total


class CollisionModule:
    def __init__(self, parameters):
        self.params = parameters
        self.groups = []
        self.group_of = {}

    @staticmethod
    def points_in_fan(points, center, ring):
        n_ring = len(ring)
        v2 = ring - center
        v3 = np.roll(ring, -1, axis=0) - center
        p = points[:, None, :] - center

        det = v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0]

        with np.errstate(divide='ignore', invalid='ignore'):
            u = (p[:, :, 0] * v3[None, :, 1] - p[:, :, 1] * v3[None, :, 0]) / det
            v = (v2[None, :, 0] * p[:, :, 1] - v2[None, :, 1] * p[:, :, 0]) / det
            inside = (u > 0) & (v > 0) & (u + v < 1)

        return inside.any(axis=1), inside.argmax(axis=1)

    def detect_and_handle(self, all_cells):
        centers = np.array([c.center_of_mass for c in all_cells])
        for cell, center in zip(all_cells, centers):
            cell.reach = np.max(np.linalg.norm(cell.m - center, axis=1))

        gaps = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        reaches = np.array([c.reach for c in all_cells])
        close = np.triu(gaps < reaches[:, None] + reaches[None, :], k=1)

        for i, j in np.argwhere(close):
            self._check_one_way(all_cells[i], all_cells[j])
            self._check_one_way(all_cells[j], all_cells[i])

    def _check_one_way(self, cell_a, cell_b):
        offsets = cell_a.m - cell_b.center_of_mass
        dists_sq = np.einsum('ij,ij->i', offsets, offsets)
        near = np.flatnonzero(~cell_a.bonded & (dists_sq < cell_b.reach ** 2))

        if near.size == 0:
            return

        inside, triangle = self.points_in_fan(cell_a.m[near],
                                              cell_b.center_of_mass,
                                              cell_b.m)
        n_a = len(cell_a.m)
        n_b = len(cell_b.m)

        for node_a, tri in zip(near[inside], triangle[inside]):
            if cell_a.bonded[node_a]:
                continue

            candidate_b = self._nearest_free_node(cell_a.m[node_a], cell_b, tri)
            if candidate_b is None:
                continue

            self._bond(cell_a, node_a, cell_b, candidate_b)

            for offset_a in [-1, 1]:
                nbr_a = (node_a + offset_a) % n_a
                if cell_a.bonded[nbr_a]:
                    continue
                for offset_b in [-1, 0, 1]:
                    nbr_b = (candidate_b + offset_b) % n_b
                    if cell_b.bonded[nbr_b]:
                        continue
                    if np.linalg.norm(cell_a.m[nbr_a] - cell_b.m[nbr_b]) < 0.1:
                        self._bond(cell_a, nbr_a, cell_b, nbr_b)
                        break

    def _nearest_free_node(self, point, cell_b, tri):
        n_b = len(cell_b.m)
        window = self.params.get('BOND_SEARCH_WINDOW', 40)
        idx = (tri + np.arange(-window, window + 2)) % n_b
        idx = idx[~cell_b.bonded[idx]]
        if idx.size == 0:
            return None
        d = np.linalg.norm(cell_b.m[idx] - point, axis=1)
        return int(idx[np.argmin(d)])

    def _bond(self, cell_a, node_a, cell_b, node_b):
        key_a = (cell_a.id, node_a)
        key_b = (cell_b.id, node_b)

        if key_a in self.group_of or key_b in self.group_of:
            return

        contact = 0.5 * (cell_a.m[node_a] + cell_b.m[node_b])
        group = [(cell_a, node_a), (cell_b, node_b)]
        self.groups.append(group)
        self.group_of[key_a] = group
        self.group_of[key_b] = group

        cell_a.bonded[node_a] = True
        cell_b.bonded[node_b] = True

        for cell, node in group:
            cell.m[node] = contact.copy()
            cell.previous_m[node] = contact.copy()
            cell.restart[node] = True


class Simulation:
    def __init__(self, parameters):
        self.params = parameters
        self.all_cells = []
        self.chemical_module = ChemicalSignalModule(parameters)
        self.collision_module = CollisionModule(parameters)
        self.time_step = 0
        self.h = self.params['MAX_TIME'] / self.params['N_TIMESTEPS']

    def initialize_simulation(self):
        p = self.params
        rng = np.random.default_rng(p.get('RANDOM_SEED'))

        signal_cell = Cell(0, initial_center=[0.0, 0.0], parameters=p)
        signal_cell.is_emitting_signal = True
        signal_cell.signal_start_time = 0.0
        self.all_cells.append(signal_cell)
        self.chemical_module.add_source(signal_cell)

        placed_centers = [np.array([0.0, 0.0])]
        max_attempts_per_cell = 500

        for outer_id in range(1, p['N_OUTER_CELLS'] + 1):
            for _ in range(max_attempts_per_cell):
                angle = rng.uniform(0, 2 * np.pi)
                radius = rng.uniform(p['PLACEMENT_RADIUS_MIN'], p['PLACEMENT_RADIUS_MAX'])
                candidate = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                if all(np.linalg.norm(candidate - c) >= p['MIN_CELL_SEPARATION']
                       for c in placed_centers):
                    placed_centers.append(candidate)
                    self.all_cells.append(Cell(outer_id, initial_center=candidate,
                                               parameters=p))
                    break
            else:
                raise RuntimeError(f"Could not place outer cell {outer_id}")

        print(f"Case 3 initialized: 1 signal cell + {p['N_OUTER_CELLS']} outer cells.")

    def run_one_step(self):
        current_time = self.time_step * self.h

        bonds_before = len(self.collision_module.groups)
        self.collision_module.detect_and_handle(self.all_cells)
        bonds_after = len(self.collision_module.groups)
        if bonds_after > bonds_before:
            print(f"t={current_time:.3f}: {bonds_after - bonds_before} new bonded pairs")

        forces = [self._calculate_forces(cell, current_time) for cell in self.all_cells]
        forces_m = np.stack([f[0] for f in forces])
        forces_n = np.stack([f[1] for f in forces])

        self._apply_adhesion_forces(forces_m)

        for cell in self.all_cells:
            self._apply_time_stepping(cell, forces_m[cell.id], forces_n[cell.id])
            cell.center_of_mass = cell.calculate_center_of_mass()

        self.time_step += 1

    def _calculate_forces(self, cell, t):
        p = self.params
        m, n = cell.m, cell.n

        effective_beta = p['BETA'] * p['REACTION_DAMPING_K'] ** np.count_nonzero(cell.bonded)
        # if t < 1.0:
        #     force_chemotaxis = np.zeros_like(m)
        # else:
        force_chemotaxis = effective_beta * self.chemical_module.calculate_gradient(m, t, cell.id)
        force_chemotaxis[cell.bonded] = 0.0

        force_internal = p['ALPHA'] * (n - m - cell.original_x_vectors)

        spring_prev = m - np.roll(m, 1, axis=0)
        spring_next = np.roll(m, -1, axis=0) - m
        y_next = np.roll(cell.original_y_vectors, -1, axis=0)
        force_surface = p['DELTA'] * (cell.original_y_vectors - spring_prev - y_next + spring_next)

        total_m = force_chemotaxis + force_internal + force_surface

        force_nuc_center = p['ALPHA_N'] * (cell.center_of_mass - n + cell.original_hat_x_vectors)
        force_reaction = -force_internal
        nuc_spring_prev = n - np.roll(n, 1, axis=0)
        nuc_spring_next = np.roll(n, -1, axis=0) - n
        hat_y_next = np.roll(cell.original_hat_y_vectors, -1, axis=0)
        force_nuc_surface = p.get('DELTA_N', p['DELTA']) * (
            cell.original_hat_y_vectors - nuc_spring_prev - hat_y_next + nuc_spring_next)

        total_n = force_nuc_center + force_reaction + force_nuc_surface
        return total_m, total_n

    def _apply_adhesion_forces(self, forces_m):
        for group in self.collision_module.groups:
            total_force = np.zeros(2)
            for cell, node in group:
                total_force += forces_m[cell.id, node]
            for cell, node in group:
                forces_m[cell.id, node] = total_force

    def _apply_time_stepping(self, cell, forces_m, forces_n):
        h = self.h
        if self.time_step == 0:
            new_m = cell.m + h * forces_m
            new_n = cell.n + h * forces_n
        else:
            new_m = (4 / 3) * cell.m - (1 / 3) * cell.previous_m + (2 * h / 3) * forces_m
            new_n = (4 / 3) * cell.n - (1 / 3) * cell.previous_n + (2 * h / 3) * forces_n
            if cell.restart.any():
                r = cell.restart
                new_m[r] = cell.m[r] + h * forces_m[r]
        cell.restart[:] = False

        cell.previous_m = cell.m.copy()
        cell.previous_n = cell.n.copy()
        cell.m = new_m
        cell.n = new_n


if __name__ == "__main__":
    TARGET_MAX_TIME = 20.0
    FRAMES = 200
    ANIMATION_SPEED_FACTOR = 50
    N_TIMESTEPS_CALCULATED = FRAMES * ANIMATION_SPEED_FACTOR

    N_NODES = 500

    sim_params = {
        'N_NODES': N_NODES,
        'MEMBRANE_RADIUS': 1.0,
        'NUCLEUS_RADIUS': 0.25,
        'ALPHA': 0.5,
        'DELTA': 30,
        'ALPHA_N': 100,
        'BETA': 2.5,
        'MU': 2.0,
        'A': 15.0,
        'T_EPSILON': 1e-6,
        'REACTION_DAMPING_K': 0.2 ** (1 / 30.0),
        'MAX_TIME': TARGET_MAX_TIME,
        'N_TIMESTEPS': N_TIMESTEPS_CALCULATED,
        'ANIMATION_SPEED_FACTOR': ANIMATION_SPEED_FACTOR,
        'N_OUTER_CELLS': 8,
        'PLACEMENT_RADIUS_MIN': 3.0,
        'PLACEMENT_RADIUS_MAX': 8.0,
        'MIN_CELL_SEPARATION': 2.2,
        'RANDOM_SEED': None,
    }

    sim = Simulation(sim_params)
    sim.initialize_simulation()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#001122')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    grid_res = 100
    x_grid = np.linspace(-10, 10, grid_res)
    y_grid = np.linspace(-10, 10, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    concentration_floor = 1e-10

    gradient_image = ax.imshow(
        np.full((grid_res, grid_res), concentration_floor),
        extent=[-10, 10, -10, 10],
        origin='lower',
        cmap=plt.get_cmap('viridis'),
        norm=LogNorm(vmin=1.0, vmax=1e2),
        zorder=1,
        alpha=0.8
)

    cell_plots = []
    for i, cell in enumerate(sim.all_cells):
        mem_pos = cell.get_membrane_positions()
        nuc_pos = cell.get_nucleus_positions()
        face = 'pink' if i == 0 else 'white'
        mem_fill = ax.fill(mem_pos[:, 0], mem_pos[:, 1], color=face, alpha=1, zorder=2)[0]
        nuc_fill = ax.fill(nuc_pos[:, 0], nuc_pos[:, 1], color=face, alpha=1, zorder=3)[0]
        mem_plot, = ax.plot(mem_pos[:, 0], mem_pos[:, 1], '-', color='black', zorder=4)
        nuc_plot, = ax.plot(nuc_pos[:, 0], nuc_pos[:, 1], '-', color='black', zorder=4)
        cell_plots.append({'mem': mem_plot, 'nuc': nuc_plot,
                           'mem_fill': mem_fill, 'nuc_fill': nuc_fill})

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

    def update_animation(frame):
        target_step = frame * sim_params['ANIMATION_SPEED_FACTOR']
        while sim.time_step < target_step:
            sim.run_one_step()
                    
        current_time = sim.time_step * sim.h
        Z = sim.chemical_module.calculate_concentration_field(X, Y, current_time)
        gradient_image.set_data(Z)
        gradient_image.set_data(np.clip(Z, concentration_floor, None))

        for i, cell in enumerate(sim.all_cells):
            mem_pos = cell.get_membrane_positions()
            nuc_pos = cell.get_nucleus_positions()
            cell_plots[i]['mem'].set_data(np.append(mem_pos[:, 0], mem_pos[0, 0]),
                                          np.append(mem_pos[:, 1], mem_pos[0, 1]))
            cell_plots[i]['nuc'].set_data(np.append(nuc_pos[:, 0], nuc_pos[0, 0]),
                                          np.append(nuc_pos[:, 1], nuc_pos[0, 1]))
            cell_plots[i]['mem_fill'].set_xy(mem_pos)
            cell_plots[i]['nuc_fill'].set_xy(nuc_pos)

        time_text.set_text(f'Time: {min(current_time, sim_params["MAX_TIME"]):.2f} hrs')

        return ([p['mem'] for p in cell_plots] + [p['nuc'] for p in cell_plots] +
                [p['mem_fill'] for p in cell_plots] + [p['nuc_fill'] for p in cell_plots] +
                [time_text, gradient_image])

    ani = FuncAnimation(fig, update_animation, frames=FRAMES + 1, interval=50, blit=True, repeat=False)
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("CASE3_multi_cells.mp4", writer=writer)
    print("Done!")