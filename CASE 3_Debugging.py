import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

class MembraneNode:
    """Single node on cell membrane"""

    def __init__(self, node_id, position):
        self.id = node_id
        self.position = np.array(position, dtype=float)
        self.current_force = np.zeros(2)


class NucleusNode:
    """Single node on nucleus boundary"""

    def __init__(self, node_id, position):
        self.id = node_id
        self.position = np.array(position, dtype=float)
        self.current_force = np.zeros(2)
        
class Cell:
    """Main cell class containing all node and state information."""

    def __init__(self, cell_id, initial_center, parameters):
        self.id = cell_id
        self.parameters = parameters
        self.is_emitting_signal = False
        self.signal_start_time = 0.0
        self.rho1 = parameters.get('CYTOPLASM_DENSITY', 1.2)  # Cytoplasm density
        self.rho2 = parameters.get('NUCLEUS_DENSITY', 1.4)  # Nucleus density (typically higher)

        # Initialize nodes
        self.membrane_nodes = []
        self.nucleus_nodes = []
        self._create_nodes(initial_center)

        # Store original vectors for spring calculations
        self._store_original_vectors()

        # Storage for time-stepping scheme (to update node positions)
        self.previous_m_positions = self.get_membrane_positions()
        self.previous_n_positions = self.get_nucleus_positions()

        # Center of mass
        self.center_of_mass = self.calculate_center_of_mass()

        # Collision information
        self.collided_nodes_info = {}  # Maps node_id to info about its attachment

    def _create_nodes(self, center):
        """Creates membrane and nucleus nodes in a circular arrangement."""
        n_nodes = self.parameters['N_NODES']
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

        for i, angle in enumerate(angles):
            m_pos = center + self.parameters['MEMBRANE_RADIUS'] * np.array([np.cos(angle), np.sin(angle)])
            self.membrane_nodes.append(MembraneNode(i, m_pos))
            n_pos = center + self.parameters['NUCLEUS_RADIUS'] * np.array([np.cos(angle), np.sin(angle)])
            self.nucleus_nodes.append(NucleusNode(i, n_pos))

    def _store_original_vectors(self):
        """Stores the initial relative positions for spring force calculations."""
        m_pos = self.get_membrane_positions()
        n_pos = self.get_nucleus_positions()

        self.original_x_vectors = n_pos - m_pos
        self.original_y_vectors = m_pos - np.roll(m_pos, 1, axis=0)

        initial_nucleus_center = np.mean(n_pos, axis=0)
        self.original_hat_x_vectors = n_pos - initial_nucleus_center
        self.original_hat_y_vectors = n_pos - np.roll(n_pos, 1, axis=0)

    def get_membrane_positions(self):
        """Returns a NumPy array of current membrane node positions."""
        return np.array([node.position for node in self.membrane_nodes])

    def get_nucleus_positions(self):
        """Returns a NumPy array of current nucleus node positions."""
        return np.array([node.position for node in self.nucleus_nodes])

    """ EQUATION 6: CENTER OF MASS """
    def calculate_center_of_mass_complex(self):
        m_pos = self.get_membrane_positions()  # (x_i, y_i)
        n_pos = self.get_nucleus_positions()  # (x̂_i, ŷ_i)
        N = len(m_pos)

        # Initialize components
        Mx = 0.0
        My = 0.0
        A = 0.0

        # Calculate for each triangular segment
        for i in range(N):
            i_next = (i + 1) % N

            # Current membrane and nucleus positions
            xi, yi = m_pos[i]
            xi_next, yi_next = m_pos[i_next]
            xi_hat, yi_hat = n_pos[i]
            xi_hat_next, yi_hat_next = n_pos[i_next]

            # Membrane triangle area contribution (cytoplasm region)
            # Triangle formed by: membrane[i], membrane[i+1], and a reference point
            membrane_area_term = (yi + yi_next) * (xi - xi_next)

            # Nucleus triangle area contribution
            nucleus_area_term = (yi_hat + yi_hat_next) * (xi_hat - xi_hat_next)

            # First moments Mx calculation
            # Cytoplasm contribution
            cytoplasm_Mx = (self.rho1 / 6) * ((xi - xi_next) *
                                              ((yi + 2 * yi_next) * xi_next + (2 * yi + yi_next) * xi))
            # Nucleus contribution
            nucleus_Mx = ((self.rho2 - self.rho1) / 6) * ((xi_hat - xi_hat_next) *
                                                          ((yi_hat + 2 * yi_hat_next) * xi_hat_next + (
                                                                      2 * yi_hat + yi_hat_next) * xi_hat))
            Mx += cytoplasm_Mx + nucleus_Mx
            # First moments My calculation
            # Cytoplasm contribution
            cytoplasm_My = (self.rho1 / 6) * ((xi - xi_next) *
                                              (yi ** 2 + yi * yi_next + yi_next ** 2))
            # Nucleus contribution
            nucleus_My = ((self.rho2 - self.rho1) / 6) * ((xi_hat - xi_hat_next) *
                                                          (yi_hat ** 2 + yi_hat * yi_hat_next + yi_hat_next ** 2))
            My += cytoplasm_My + nucleus_My
            # Total area calculation
            cytoplasm_area = (self.rho1 / 2) * membrane_area_term
            nucleus_area = ((self.rho2 - self.rho1) / 2) * nucleus_area_term
            A += cytoplasm_area + nucleus_area

        # Avoid division by zero
        if abs(A) < 1e-10:
            return np.mean(m_pos, axis=0)
        # Center of mass coordinates
        x_center = Mx / A
        y_center = My / A
        return np.array([x_center, y_center])

    def calculate_center_of_mass(self):
        return self.calculate_center_of_mass_complex()
    
class ChemicalSignalModule:
    """A module to handle chemical signal calculations (Eq. 8 & 9)."""

    def __init__(self, parameters):
        self.sources = []  # List of emitting cells
        self.params = parameters

    def add_source(self, cell):
        self.sources.append(cell)

    def calculate_gradient(self, position, t): #equation 9
        """Calculates the chemical gradient at a given position and time."""
        total_gradient = np.zeros(2)
        for source_cell in self.sources:
            if not source_cell.is_emitting_signal or t < source_cell.signal_start_time:
                continue

            effective_time = t - source_cell.signal_start_time + self.params['T_EPSILON']
            source_pos = source_cell.center_of_mass
            vec_diff = position - source_pos
            dist_sq = np.sum(vec_diff ** 2)

            exp_term = np.exp(-dist_sq / (4 * self.params['MU'] * effective_time))
            coefficient = -self.params['A'] / (2 * self.params['MU'] ** 2 * effective_time ** 2)
            total_gradient += coefficient * vec_diff * exp_term
        return total_gradient

    def calculate_concentration_field(self, X, Y, t): #equation 8
        """Calculates the concentration across a grid for visualization."""
        total_concentration = np.zeros_like(X)
        for source_cell in self.sources:
            if not source_cell.is_emitting_signal or t < source_cell.signal_start_time:
                continue

            effective_time = t - source_cell.signal_start_time + self.params['T_EPSILON']
            source_pos = source_cell.center_of_mass
            dist_sq = (X - source_pos[0]) ** 2 + (Y - source_pos[1]) ** 2

            denominator = self.params['MU'] * effective_time
            concentration = (self.params['A'] / denominator) * np.exp(-dist_sq / (4 * denominator))
            total_concentration += np.nan_to_num(concentration)
        return total_concentration

    def calculate_gradient_excluding_cell(self, position, t, excluding_cell_id):
        """Signal emitting source_pos from center of mass"""
        total_gradient = np.zeros(2)
        for source_cell in self.sources:
            if (not source_cell.is_emitting_signal or
                    t < source_cell.signal_start_time or
                    source_cell.id == excluding_cell_id):  # Skip the excluded cell
                continue

            effective_time = t - source_cell.signal_start_time + self.params['T_EPSILON']
            source_pos = source_cell.center_of_mass
            vec_diff = position - source_pos
            dist_sq = np.sum(vec_diff ** 2)

            exp_term = np.exp(-dist_sq / (4 * self.params['MU'] * effective_time))
            coefficient = -self.params['A'] / (2 * self.params['MU'] ** 2 * effective_time ** 2)
            total_gradient += coefficient * vec_diff * exp_term
        return total_gradient

class CollisionModule:
    """A module for collision detection and handling."""

    def is_point_in_triangle(self, point, v1, v2, v3):
        """Checks if a point is inside a triangle using barycentric coordinates."""
        d0, d1, d2 = v3 - v1, v2 - v1, point - v1
        dot00, dot01, dot02 = np.dot(d0, d0), np.dot(d0, d1), np.dot(d0, d2)
        dot11, dot12 = np.dot(d1, d1), np.dot(d1, d2)
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        return (u > 0) and (v > 0) and (u + v < 1)

    # u and v

    def detect_and_handle(self, all_cells):
        """Detects and handles collisions between all pairs of cells."""
        for i in range(len(all_cells)):
            for j in range(i + 1, len(all_cells)): #avoid duplicates
                cell_a, cell_b = all_cells[i], all_cells[j]
                self._check_one_way_collision(cell_a, cell_b)
                self._check_one_way_collision(cell_b, cell_a)

    def _check_one_way_collision(self, cell_a, cell_b):
        """Checks if any node of cell_a has collided with cell_b."""
        center_b = cell_b.center_of_mass
        m_pos_b = cell_b.get_membrane_positions()

        for node_a_idx, node_a in enumerate(cell_a.membrane_nodes):
            if node_a_idx in cell_a.collided_nodes_info: continue

            for node_b_idx in range(len(cell_b.membrane_nodes)):
                v1 = center_b #nucleus center
                v2 = m_pos_b[node_b_idx] #current membrane node
                v3 = m_pos_b[(node_b_idx + 1) % len(cell_b.membrane_nodes)] #next membrane node

                if self.is_point_in_triangle(node_a.position, v1, v2, v3):
                    # Attach to the closest vertex
                    dist_v2 = np.linalg.norm(node_a.position - v2)
                    dist_v3 = np.linalg.norm(node_a.position - v3)
                    closest_b_idx = (
                        node_b_idx 
                        if dist_v2 < dist_v3 
                        else (node_b_idx + 1) % len(cell_b.membrane_nodes)
                    )
                    node_b = cell_b.membrane_nodes[closest_b_idx]

                    contact_position = (node_a.position + node_b.position) / 2
                    node_a.position = contact_position.copy()
                    node_b.position = contact_position.copy()

                    cell_a.previous_m_positions[node_a_idx] = contact_position.copy()
                    cell_b.previous_m_positions[closest_b_idx] = contact_position.copy()
                    
                    cell_a.collided_nodes_info[node_a_idx] = {'partner_cell': cell_b, 'partner_node_idx': closest_b_idx}
                    cell_b.collided_nodes_info[closest_b_idx] = {'partner_cell': cell_a, 'partner_node_idx': node_a_idx}
                    print(f"Collision Locked: Cell {cell_a.id}[{node_a_idx}] <-> Cell {cell_b.id}[{closest_b_idx}]")
                    break
                
class Simulation:
    """Main class to control the entire simulation flow."""

    def __init__(self, parameters):
        self.params = parameters
        self.all_cells = []
        self.chemical_module = ChemicalSignalModule(parameters)
        self.collision_module = CollisionModule()
        self.time_step = 0
        self.h = self.params['MAX_TIME'] / self.params['N_TIMESTEPS']

    def initialize_simulation(self):
        """Loads parameters and creates initial cell configurations."""
        p = self.params
        rng = np.random.default_rng(p.get('RANDOM_SEED'))

        # Create a stationary, signal-emitting cell
        signal_cell = Cell(0, initial_center=[0.0, 0.0], parameters=p)
        signal_cell.is_emitting_signal = True
        signal_cell.signal_start_time = 1.0  # Start signal at t=1
        self.all_cells.append(signal_cell)
        self.chemical_module.add_source(signal_cell)

        placed_centers = [np.array([0.0, 0.0])]  # include signal cell center
        n_outer = p['N_OUTER_CELLS']
        r_min = p['PLACEMENT_RADIUS_MIN']
        r_max = p['PLACEMENT_RADIUS_MAX']
        min_sep = p['MIN_CELL_SEPARATION']
        max_attempts_per_cell = 500

        outer_id = 1
        while outer_id <= n_outer:
            placed = False
            for _attempt in range(max_attempts_per_cell):
                # random angle, random radius in [r_min, r_max]
                angle = rng.uniform(0, 2 * np.pi)
                radius = rng.uniform(r_min, r_max)
                candidate = np.array([radius * np.cos(angle), radius * np.sin(angle)])

                # reject if too close to any already-placed cell (including center)
                if all(np.linalg.norm(candidate - c) >= min_sep for c in placed_centers):
                    placed_centers.append(candidate)
                    outer_cell = Cell(outer_id, initial_center=candidate, parameters=p)
                    self.all_cells.append(outer_cell)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Could not place outer cell {outer_id} without overlap after "
                    f"{max_attempts_per_cell} attempts. Consider lowering "
                    f"MIN_CELL_SEPARATION or widening the placement radius range."
                )
            outer_id += 1

        print(f"Case 3 initialized: 1 signal cell + {n_outer} outer cells "
              f"(seed={p.get('RANDOM_SEED')}).")
        print("Outer cell centers:")
        for c in self.all_cells[1:]:
            print(f"  Cell {c.id}: {c.get_membrane_positions().mean(axis=0).round(2)}")

            # # Create a moving cell
            # moving_cell = Cell(1, initial_center=[-2, 0], parameters=self.params)
            # self.all_cells.append(moving_cell)

    def run_one_step(self):
        """Executes one full time step of the simulation."""
        current_time = self.time_step * self.h

        # 1. Update Chemical Signal (gradient will be calculated as needed)

        # 2. Detect and Handle Collisions
        self.collision_module.detect_and_handle(self.all_cells)

        # 3. Calculate forces for all cells
        all_forces_m = {cell.id: np.zeros_like(cell.get_membrane_positions()) for cell in self.all_cells}
        all_forces_n = {cell.id: np.zeros_like(cell.get_nucleus_positions()) for cell in self.all_cells}

        for cell in self.all_cells:
            m_forces, n_forces = self._calculate_forces(cell, current_time)
            all_forces_m[cell.id] = m_forces
            all_forces_n[cell.id] = n_forces

        # 4. Handle force summation for collided nodes (Equation 10)
        self._apply_adhesion_forces(all_forces_m)

        # 5. Apply time-stepping to update positions
        for cell in self.all_cells:
            self._apply_time_stepping(cell, all_forces_m[cell.id], all_forces_n[cell.id])
            cell.center_of_mass = cell.calculate_center_of_mass()  # Update center of mass from Eq 11

        self.time_step += 1

    def _calculate_forces(self, cell, t):
        """Calculates the forces acting on a single cell's nodes."""
        p = self.params
        m_pos = cell.get_membrane_positions()
        n_pos = cell.get_nucleus_positions()

        # Membrane Forces (Eq. 1, 2, 3)

        num_attached = len(cell.collided_nodes_info)
        effective_beta = p['BETA'] * (p['REACTION_DAMPING_K'] ** num_attached)

        if t < 1.0:  # Signal starts at t=1.0
            force_chemotaxis = np.zeros_like(m_pos)
        elif cell.is_emitting_signal:
            grad = np.array([self.chemical_module.calculate_gradient_excluding_cell(pos, t, cell.id)
                             for pos in m_pos])
            force_chemotaxis = effective_beta * grad
        else:
            # Non-signal cells respond to all gradients
            grad = np.array([self.chemical_module.calculate_gradient(pos, t) for pos in m_pos])
            force_chemotaxis = effective_beta * grad #Equation 1

        force_internal = p['ALPHA'] * (n_pos - m_pos - cell.original_x_vectors) #Equation 2

        # External membrane relaxation force
        m_prev = np.roll(m_pos, 1, axis=0)
        m_next = np.roll(m_pos, -1, axis=0)
        spring_prev = m_pos - m_prev #clockwise direction
        spring_next = m_next - m_pos
        y_next = np.roll(cell.original_y_vectors, -1, axis=0)
        force_surface = p['DELTA'] * (cell.original_y_vectors - spring_prev - y_next + spring_next) #Equation 3

        total_m_force = force_chemotaxis + force_internal + force_surface #Equation 4

        # Nucleus Forces (Eq. 5)
        #nucleus_center = np.mean(n_pos, axis=0)
        nucleus_center = cell.center_of_mass
        force_nuc_center = p['ALPHA_N'] * (nucleus_center - n_pos + cell.original_hat_x_vectors)
        force_reaction = -force_internal

        n_prev = np.roll(n_pos, 1, axis=0)
        n_next = np.roll(n_pos, -1, axis=0)
        nuc_spring_prev = n_pos - n_prev
        nuc_spring_next = n_next - n_pos
        hat_y_next = np.roll(cell.original_hat_y_vectors, -1, axis=0)
        force_nuc_surface = p['DELTA'] * (cell.original_hat_y_vectors - nuc_spring_prev - hat_y_next + nuc_spring_next)

        total_n_force = force_nuc_center + force_reaction + force_nuc_surface #Equation 5

        return total_m_force, total_n_force

    # EQUATION 10 and 11

    def _apply_adhesion_forces(self, all_forces_m):
        """Sum forces for attached nodes according to Eq. 10."""
        processed_pairs = set()
        for cell_a in self.all_cells:
            for node_a_idx, info in cell_a.collided_nodes_info.items():
                cell_b = info['partner_cell']
                node_b_idx = info['partner_node_idx']

                # Create a unique key for the pair to avoid double processing
                pair_key = tuple(sorted(((cell_a.id, node_a_idx), (cell_b.id, node_b_idx))))
                if pair_key in processed_pairs:
                    continue

                force_a = all_forces_m[cell_a.id][node_a_idx]
                force_b = all_forces_m[cell_b.id][node_b_idx]
                combined_force = force_a + force_b

                all_forces_m[cell_a.id][node_a_idx] = combined_force
                all_forces_m[cell_b.id][node_b_idx] = combined_force
                processed_pairs.add(pair_key)

    def _apply_time_stepping(self, cell, m_forces, n_forces):
        """Updates node positions using (Eq. 11)."""
        m_pos = cell.get_membrane_positions()
        n_pos = cell.get_nucleus_positions()

        if self.time_step == 0:  # Euler's method for first step
            new_m_pos = m_pos + self.h * m_forces
            new_n_pos = n_pos + self.h * n_forces
        else:
            new_m_pos = (4 / 3) * m_pos - (1 / 3) * cell.previous_m_positions + (2 * self.h / 3) * m_forces
            new_n_pos = (4 / 3) * n_pos - (1 / 3) * cell.previous_n_positions + (2 * self.h / 3) * n_forces

        cell.previous_m_positions = m_pos.copy()
        cell.previous_n_positions = n_pos.copy()

        # Update node objects
        for i, node in enumerate(cell.membrane_nodes): node.position = new_m_pos[i]
        for i, node in enumerate(cell.nucleus_nodes): node.position = new_n_pos[i]
        
import numpy as np
if __name__ == "__main__":
    # --- More Intuitive Parameter Setup ---
    TARGET_MAX_TIME = 20.0
    FRAMES = 200  # The number of frames you want to see
    ANIMATION_SPEED_FACTOR = 50 #How many simulation steps per frame
    #total of 10,000 number of steps

    # Derive the simulation parameters from the desired output
    N_TIMESTEPS_CALCULATED = FRAMES * ANIMATION_SPEED_FACTOR #10.000

    sim_params = {
        'N_NODES': 100, #100,
        'MEMBRANE_RADIUS': 1.0,
        'NUCLEUS_RADIUS': 0.25,
        'ALPHA': 0.5, 'DELTA': 30, 'ALPHA_N': 100, 'BETA': 2.5,'MU': 2.0, 'A': 15.0, 'T_EPSILON': 1e-6,
        'REACTION_DAMPING_K': 0.9227,
        'MAX_TIME': TARGET_MAX_TIME,
        'N_TIMESTEPS': N_TIMESTEPS_CALCULATED,
        'ANIMATION_SPEED_FACTOR': ANIMATION_SPEED_FACTOR,
        'N_OUTER_CELLS': 8,
        'PLACEMENT_RADIUS_MIN': 4.0,
        'PLACEMENT_RADIUS_MAX': 8.0,
        'MIN_CELL_SEPARATION': 2.2,
        'RANDOM_SEED': None,
    }

    # --- Setup Simulation ---
    sim = Simulation(sim_params)
    sim.initialize_simulation()

    # --- Setup Visualization ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#001122')
    ax.set_aspect('equal', 'box')
    #ax.set_title("Cell Migration Simulation", color='black')
    ax.set_xlim(-10, 10);
    ax.set_ylim(-10, 10)

    # Concentration field
    grid_res = 100
    x_grid = np.linspace(-10, 10, grid_res)
    y_grid = np.linspace(-10, 10, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    cmap = plt.get_cmap('viridis')
    gradient_image = ax.imshow(np.zeros((grid_res, grid_res)), extent=[-10, 10, -10, 10],
                               origin='lower', cmap=cmap, zorder=1, alpha=0.8)

    # Cell patches
    cell_plots = []
    #colors = ['darkred', 'darkblue']
    #fill_colors = ['lightgray', 'skyblue']
    for i, cell in enumerate(sim.all_cells):
        mem_pos = cell.get_membrane_positions()
        nuc_pos = cell.get_nucleus_positions()

        # Fill the membrane area
        mem_fill = ax.fill(mem_pos[:, 0], mem_pos[:, 1], color='pink' if i==0 else 'white', alpha=1, zorder=2)[0]

        # Fill the nucleus area
        nuc_fill = ax.fill(nuc_pos[:, 0], nuc_pos[:, 1], color='pink' if i==0 else 'white', alpha=1, zorder=3)[0]

        # Draw the boundaries
        mem_plot, = ax.plot(mem_pos[:, 0], mem_pos[:, 1], '-o', ms=2, color='black', zorder=4)
        nuc_plot, = ax.plot(nuc_pos[:, 0], nuc_pos[:, 1], '-', color='black', zorder=4)

        cell_plots.append({
            'mem': mem_plot,
            'nuc': nuc_plot,
            'mem_fill': mem_fill,
            'nuc_fill': nuc_fill
        })

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')


    def update_animation(frame):
        for _ in range(sim_params['ANIMATION_SPEED_FACTOR']):
            sim.run_one_step()

        # Update visualization
        current_time = sim.time_step * sim.h

        Z = sim.chemical_module.calculate_concentration_field(X, Y, current_time)
        gradient_image.set_data(Z)
        if frame < 5:  # Auto-adjust color limits for the first few frames
            gradient_image.set_clim(vmin=0, vmax=np.max(Z) if np.max(Z) > 0 else 1.0)

        for i, cell in enumerate(sim.all_cells):
            mem_pos = cell.get_membrane_positions()
            nuc_pos = cell.get_nucleus_positions()
            cell_plots[i]['mem'].set_data(np.append(mem_pos[:, 0], mem_pos[0, 0]),
                                          np.append(mem_pos[:, 1], mem_pos[0, 1]))
            cell_plots[i]['nuc'].set_data(np.append(nuc_pos[:, 0], nuc_pos[0, 0]),
                                          np.append(nuc_pos[:, 1], nuc_pos[0, 1]))
            # Update fill areas
            cell_plots[i]['mem_fill'].set_xy(mem_pos)
            cell_plots[i]['nuc_fill'].set_xy(nuc_pos)

        display_time = min(current_time, sim_params['MAX_TIME'])
        time_text.set_text(f'Time: {display_time:.2f} hrs')

        artists = ([p['mem'] for p in cell_plots] + [p['nuc'] for p in cell_plots] +
                   [p['mem_fill'] for p in cell_plots] + [p['nuc_fill'] for p in cell_plots] +
                   [time_text, gradient_image])
        return artists


    ani = FuncAnimation(fig, update_animation, frames=FRAMES, interval=50, blit=True, repeat=False)
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("CASE 3 (100 nodes).mp4", writer="ffmpeg", fps=20)
    print("Done!")                