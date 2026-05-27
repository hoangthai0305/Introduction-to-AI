import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from abc import ABC, abstractmethod

class ObjectiveFunction:
    def __call__(self, x, y):
        return self.evaluate(x, y)

    def evaluate(self, x, y):
        return np.sin(x/8) + np.cos(y/4) - np.sin(x*y/16) + np.cos((x**2)/16) + np.sin((y**2)/8)

class Problem(ABC):
    @property
    @abstractmethod
    def initial_state(self):
        pass

    @abstractmethod
    def goal_test(self, state):
        pass

    @abstractmethod
    def actions(self, state):
        pass

    @abstractmethod
    def result(self, state, action):
        pass

    @abstractmethod
    def value(self, state):
        pass

class SASearchProblem(Problem):
    def __init__(self, objective_function, initial_state=(0, 0), step_size=np.pi/32):
        self._initial_state = initial_state
        self.step_size = step_size
        self.objective_function = objective_function

    @property
    def initial_state(self):
        return self._initial_state

    def goal_test(self, state):
        return False

    def actions(self, state):
        moves = []
        for dx in [0, self.step_size, -self.step_size]:
            for dy in [0, self.step_size, -self.step_size]:
                moves.append((dx, dy))
        return moves

    def result(self, state, action):
        x, y = state
        dx, dy = action
        return (x + dx, y + dy)

    def value(self, state):
        x, y = state
        return self.objective_function(x, y)

class Schedule(ABC):
    @abstractmethod
    def temperature(self, t):
        pass

class LinearSchedule(Schedule):
    def __init__(self, initial_temp = 1.0, cooling_rate = 0.1):
        if initial_temp < 0:
            raise ValueError("Initial temperature cannot be negative.")
        if cooling_rate <= 0:
            raise ValueError("Cooling rate must be positive for temperature to decrease.")
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def temperature(self, t):
        current_temp = self.initial_temp - (self.cooling_rate * t)
        return max(0.0, current_temp)

class SimulatedAnnealingSearch:
    def __init__(self, problem, schedule):
        self.problem = problem
        self.schedule = schedule
        self.current_state = None
        self.current_value = None
        self.best_state = None
        self.best_value = None
        self.path = []
        self.temperatures_log = []
        self.scores_log = []

    def search(self, max_iterations = 1000):
        for t in range(max_iterations):
            temperature = self.schedule.temperature(t)

            if temperature < 1e-10: # Temperature is effectively zero
                break

            self.temperatures_log.append(temperature)

            actions = self.problem.actions(self.current_state)

            if not actions:
                break
            action = random.choice(actions)

            next_state = self.problem.result(self.current_state, action)
            next_value = self.problem.value(next_state)

            if self._accept_transition(self.current_value, next_value, temperature):
                self.current_state = next_state
                self.current_value = next_value

            self.path.append(self.current_state)
            self.scores_log.append(self.current_value)

            if self.current_value > self.best_value:
                self.best_state = self.current_state
                self.best_value = self.current_value

        return self.best_state, self.best_value

    def _accept_transition(self, current_value, next_value, temperature):
        if next_value > current_value:
            return True
        delta = next_value - current_value
        if temperature == 0:
            return False
        probability = math.exp(delta / temperature)
        return random.random() < probability

    def get_search_path(self):
        return self.path

    def get_temperatures_log(self):
        return self.temperatures_log

    def get_scores_log(self):
        return self.scores_log

class SurfaceVisualizer:
    def __init__(self):
        self.fig_3d = plt.figure("3D Surface Plot", figsize=(10, 8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        self.fig_2d = plt.figure("2D Contour Plot", figsize=(10, 8))
        self.ax_2d = self.fig_2d.add_subplot(111)

        self.fig_stats = plt.figure("Statistics Plot", figsize=(10, 6))
        self.ax_stats = self.fig_stats.add_subplot(111)

    def plot_surface_3d(self, function, x_range, y_range, resolution = 100):
        x_coords = np.linspace(x_range[0], x_range[1], resolution)
        y_coords = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)

        Z_values = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_values[i, j] = function(X[i, j], Y[i, j])

        self.ax_3d.plot_surface(X, Y, Z_values, cmap='viridis', alpha=1, edgecolor='none', zorder=1)
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('f(X, Y)')
        self.ax_3d.set_title('3D Surface Plot with Search Path')
        self.ax_3d.set_xlim(x_range)
        self.ax_3d.set_ylim(y_range)
        self.ax_3d.invert_xaxis()


    def plot_path_3d(self, path, function, color = 'red', marker = 'o', linewidth = 1, markersize = 0.5):
        if not path or len(path) < 1:
            return

        x_vals = [p[0] for p in path]
        y_vals = [p[1] for p in path]
        z_vals_on_surface = [function(p[0], p[1]) for p in path]

        xlim = self.ax_3d.get_xlim()
        ylim = self.ax_3d.get_ylim()

        x_sample = np.linspace(xlim[0], xlim[1], 20)
        y_sample = np.linspace(ylim[0], ylim[1], 20)
        X_sample, Y_sample = np.meshgrid(x_sample, y_sample)

        Z_sample_values_list = []
        for r_idx in range(X_sample.shape[0]):
            for c_idx in range(X_sample.shape[1]):
                 Z_sample_values_list.append(function(X_sample[r_idx, c_idx], Y_sample[r_idx, c_idx]))
        Z_sample_np = np.array(Z_sample_values_list)

        if Z_sample_np.size > 0:
            z_range_val = np.ptp(Z_sample_np)
            z_offset = z_range_val * 0.03 if z_range_val > 1e-6 else 0.15
        else:
            z_range_path_val = np.ptp(np.array(z_vals_on_surface)) if len(z_vals_on_surface) > 1 else 1.0
            z_offset = z_range_path_val * 0.03 if z_range_path_val > 1e-6 else 0.15

        z_vals_plot = [z + z_offset for z in z_vals_on_surface]

        self.ax_3d.plot(x_vals, y_vals, z_vals_plot, color=color, marker=marker, linewidth=linewidth, markersize=markersize if markersize > 0 else None, zorder=10, label='Search Path')

        start_coord = path[0]
        start_z_on_surface = z_vals_on_surface[0]
        start_z_plot = start_z_on_surface + z_offset*4
        self.ax_3d.scatter(start_coord[0], start_coord[1], start_z_plot, color='cyan', s=250, marker='P', label=f'Start ({start_z_on_surface:.2f})', depthshade=True, edgecolor='black', linewidth=2.0, zorder=12)

        if len(path) > 1:
            end_coord = path[-1]
            end_z_on_surface = z_vals_on_surface[-1]
            end_z_plot = end_z_on_surface + z_offset*4
            self.ax_3d.scatter(end_coord[0], end_coord[1], end_z_plot, color='magenta', s=250, marker='X', label=f'End ({end_z_on_surface:.2f})', depthshade=True, edgecolor='black', linewidth=2.0, zorder=11)

        best_state_in_path = path[0]
        best_value_in_path = function(*best_state_in_path)
        for p_coord in path:
            val = function(*p_coord)
            if val > best_value_in_path:
                best_value_in_path = val
                best_state_in_path = p_coord

        best_z_on_surface = best_value_in_path
        best_z_plot = best_z_on_surface + z_offset*4
        self.ax_3d.scatter(best_state_in_path[0], best_state_in_path[1], best_z_plot, color='yellow', s=350, marker='*', label=f'Best in Path ({best_z_on_surface:.2f})', depthshade=True, edgecolor='black', linewidth=2.0, zorder=13)

        self.ax_3d.legend(loc='upper left')
        self.fig_3d.tight_layout()

    def plot_contour_2d(self, function, x_range, y_range, resolution = 100):
        x_coords = np.linspace(x_range[0], x_range[1], resolution)
        y_coords = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)

        Z_values = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_values[i, j] = function(X[i, j], Y[i, j])

        cp = self.ax_2d.contourf(X, Y, Z_values, 20, cmap='viridis')
        self.fig_2d.colorbar(cp, ax=self.ax_2d)
        self.ax_2d.set_xlabel('X')
        self.ax_2d.set_ylabel('Y')
        self.ax_2d.set_title('2D Contour Plot with Search Path')
        self.ax_2d.set_xlim(x_range)
        self.ax_2d.set_ylim(y_range)


    def plot_path_2d(self, path, function, color = 'red', marker = 'o', linewidth = 1, markersize = 0.5):
        if not path or len(path) < 1:
            return

        x_vals = [p[0] for p in path]
        y_vals = [p[1] for p in path]

        self.ax_2d.plot(x_vals, y_vals, color=color, marker=marker, markersize=markersize if markersize > 0 else None, linewidth=linewidth, label='Search Path')

        start_coord = path[0]
        self.ax_2d.scatter(start_coord[0], start_coord[1], color='cyan', s=200, marker='P', label=f'Start ({function(*start_coord):.2f})', edgecolor='black', linewidth=2.0, zorder=5)

        if len(path) > 1:
            end_coord = path[-1]
            self.ax_2d.scatter(end_coord[0], end_coord[1], color='magenta', s=200, marker='X', label=f'End ({function(*end_coord):.2f})', edgecolor='black', linewidth=2.0, zorder=5)

        best_state_in_path = path[0]
        best_value_in_path = function(*best_state_in_path)
        for p_coord in path:
            val = function(*p_coord)
            if val > best_value_in_path:
                best_value_in_path = val
                best_state_in_path = p_coord
        self.ax_2d.scatter(best_state_in_path[0], best_state_in_path[1], color='yellow', s=250, marker='*', label=f'Best in Path ({best_value_in_path:.2f})', edgecolor='black', linewidth=2.0, zorder=6)

        self.ax_2d.legend(loc='upper left')
        self.fig_2d.tight_layout()

    def plot_stats(self, scores_log, temperatures_log):
        actual_iterations_run = len(temperatures_log)

        if actual_iterations_run == 0:
            self.ax_stats.text(0.5, 0.5, "No iterations to plot stats for (0 iterations run).", ha='center', va='center', transform=self.ax_stats.transAxes)
            self.fig_stats.tight_layout()
            return

        iteration_labels_for_axis = range(actual_iterations_run)
        # Ensure scores_log has at least one element before slicing beyond the first
        scores_at_end_of_iter_t = scores_log[1 : actual_iterations_run + 1] if len(scores_log) > 0 else []
        temps_at_start_of_iter_t = temperatures_log[:actual_iterations_run]

        color_score = 'darkgreen'
        self.ax_stats.set_xlabel('Iteration Index (t)', fontsize=12)
        self.ax_stats.set_ylabel('Score (at end of iteration t)', color=color_score, fontsize=12)
        if list(iteration_labels_for_axis) and scores_at_end_of_iter_t: # Check if scores list is not empty
            self.ax_stats.plot(iteration_labels_for_axis, scores_at_end_of_iter_t, color=color_score, linestyle='-', marker='.', markersize=5, alpha=0.8, label='Score')

        self.ax_stats.tick_params(axis='y', labelcolor=color_score, labelsize=10)
        self.ax_stats.tick_params(axis='x', labelsize=10)
        self.ax_stats.grid(True, linestyle=':', alpha=0.7)

        ax_temp = self.ax_stats.twinx()
        color_temp = 'darkblue'
        ax_temp.set_ylabel('Temperature (at start of iteration t)', color=color_temp, fontsize=12)
        if list(iteration_labels_for_axis):
            ax_temp.plot(iteration_labels_for_axis, temps_at_start_of_iter_t, color=color_temp, linestyle='--', marker='x', markersize=5, alpha=0.8, label='Temperature')
        ax_temp.tick_params(axis='y', labelcolor=color_temp, labelsize=10)
        ax_temp.set_ylim(bottom=0) # Ensure temperature doesn't go below zero on plot

        handles, labels = self.ax_stats.get_legend_handles_labels()
        handles2, labels2 = ax_temp.get_legend_handles_labels()
        if handles or handles2:
            ax_temp.legend(handles + handles2, labels + labels2, loc='center right', fontsize=10)

        self.ax_stats.set_title('Score & Temperature vs. Iterations', fontsize=14)
        self.fig_stats.tight_layout()

    def display(self):
        plt.show()

    def save_all_figures(self, base_filename = "run"):
        try:
            fn_3d = f"{base_filename}_3d_surface.png"
            self.fig_3d.savefig(fn_3d)
            print(f"Saved 3D plot to {fn_3d}")
        except Exception as e:
            print(f"Could not save 3D plot: {e}")
        try:
            fn_2d = f"{base_filename}_2d_contour.png"
            self.fig_2d.savefig(fn_2d)
            print(f"Saved 2D contour plot to {fn_2d}")
        except Exception as e:
            print(f"Could not save 2D plot: {e}")
        try:
            fn_stats = f"{base_filename}_stats.png"
            self.fig_stats.savefig(fn_stats)
            print(f"Saved Statistics plot to {fn_stats}")
        except Exception as e:
            print(f"Could not save Stats plot: {e}")

class SimulatedAnnealingApp:
    def __init__(self, objective_function, initial_state = (0, 0),
                 step_size = np.pi/32, schedule = None, run_id = "default"):
        self.objective_function = objective_function
        self.problem = SASearchProblem(objective_function, initial_state, step_size)
        self.run_id = run_id

        if schedule is None:
            schedule = LinearSchedule(initial_temp=1000.0, cooling_rate=1.0) # Default schedule

        self.searcher = SimulatedAnnealingSearch(self.problem, schedule)
        self.visualizer = SurfaceVisualizer()

    def run(self, max_iterations = 1000):
        initial_score = self.problem.value(self.problem.initial_state)

        self.searcher.current_state = self.problem.initial_state
        self.searcher.current_value = initial_score
        self.searcher.best_state = self.problem.initial_state
        self.searcher.best_value = initial_score

        self.searcher.path = [self.problem.initial_state]
        self.searcher.scores_log = [initial_score] # Initial score for iteration 0 (before first step)
        self.searcher.temperatures_log = [] # Temperatures are logged at the start of each iteration t

        try:
            if self.visualizer.fig_3d.canvas.manager:
                self.visualizer.fig_3d.canvas.manager.set_window_title(f"3D Surface - {self.run_id} ({self.searcher.schedule.__class__.__name__})")
            if self.visualizer.fig_2d.canvas.manager:
                self.visualizer.fig_2d.canvas.manager.set_window_title(f"2D Contour - {self.run_id} ({self.searcher.schedule.__class__.__name__})")
            if self.visualizer.fig_stats.canvas.manager:
                self.visualizer.fig_stats.canvas.manager.set_window_title(f"Statistics - {self.run_id} ({self.searcher.schedule.__class__.__name__})")
        except AttributeError:
            pass

        best_state_overall, best_value_overall = self.searcher.search(max_iterations)

        print(f"\n--- Search Run ID: {self.run_id} with {self.searcher.schedule.__class__.__name__} ---")
        print(f"Initial state: {self.problem.initial_state}, Value: {self.objective_function(*self.problem.initial_state):.4f}")
        print(f"Best solution found by SA: ({self.searcher.best_state[0]:.4f}, {self.searcher.best_state[1]:.4f})")
        print(f"Value at best SA solution: {self.searcher.best_value:.4f}")
        print(f"Last visited state: ({self.searcher.current_state[0]:.4f}, {self.searcher.current_state[1]:.4f}), Value: {self.searcher.current_value:.4f}")
        actual_iterations_run = len(self.searcher.get_temperatures_log())
        print(f"Target max_iterations: {max_iterations}, Actual iterations run: {actual_iterations_run}")
        if self.searcher.get_temperatures_log():
             print(f"Final temperature: {self.searcher.get_temperatures_log()[-1]:.4e}")
        else:
             print(f"Final temperature: N/A (0 iterations run)")
        return best_state_overall, best_value_overall

    def visualize_results(self, x_range_user = None,
                          y_range_user = None,
                          padding_factor = 0.2,
                          min_view_size = 7.0):

        path_coords = self.searcher.get_search_path()

        if not path_coords:
            final_x_range = x_range_user if x_range_user is not None else (-min_view_size/2, min_view_size/2)
            final_y_range = y_range_user if y_range_user is not None else (-min_view_size/2, min_view_size/2)
        else:
            x_coords_list = [p[0] for p in path_coords]
            y_coords_list = [p[1] for p in path_coords]

            min_x_path, max_x_path = min(x_coords_list), max(x_coords_list)
            min_y_path, max_y_path = min(y_coords_list), max(y_coords_list)

            path_width = max_x_path - min_x_path
            path_height = max_y_path - min_y_path

            x_padding = path_width * padding_factor
            y_padding = path_height * padding_factor

            calc_min_x = min_x_path - x_padding
            calc_max_x = max_x_path + x_padding
            calc_min_y = min_y_path - y_padding
            calc_max_y = max_y_path + y_padding

            current_width = calc_max_x - calc_min_x
            if current_width < min_view_size:
                center_x = (calc_min_x + calc_max_x) / 2
                calc_min_x = center_x - min_view_size / 2
                calc_max_x = center_x + min_view_size / 2

            current_height = calc_max_y - calc_min_y
            if current_height < min_view_size:
                center_y = (calc_min_y + calc_max_y) / 2
                calc_min_y = center_y - min_view_size / 2
                calc_max_y = center_y + min_view_size / 2

            final_x_range = x_range_user if x_range_user is not None else (calc_min_x, calc_max_x)
            final_y_range = y_range_user if y_range_user is not None else (calc_min_y, calc_max_y)

        self.visualizer.plot_surface_3d(self.objective_function, final_x_range, final_y_range)
        self.visualizer.plot_path_3d(self.searcher.get_search_path(), self.objective_function, markersize=0.1)

        self.visualizer.plot_contour_2d(self.objective_function, final_x_range, final_y_range)
        self.visualizer.plot_path_2d(self.searcher.get_search_path(), self.objective_function, markersize=0.1)

        self.visualizer.plot_stats(self.searcher.get_scores_log(), self.searcher.get_temperatures_log())


if __name__ == "__main__":
    random.seed(99)
    objective_function = ObjectiveFunction()

    initial_temp_param = 5000.0
    step_s = np.pi/32
    initial_coords = (0.0, 0.0)

    print("Running with Linear Schedule:")
    linear_cooling_rate = 0.5
    # Calculate iterations needed for temperature to reach near zero
    linear_max_iters_target = int(initial_temp_param / linear_cooling_rate) + 50 # Add a small buffer

    linear_schedule_instance = LinearSchedule(initial_temp=initial_temp_param, cooling_rate=linear_cooling_rate)
    app_linear = SimulatedAnnealingApp(
        objective_function=objective_function,
        initial_state=initial_coords,
        step_size=step_s,
        schedule=linear_schedule_instance,
        run_id="linearschedule"
    )

    app_linear.run(max_iterations=linear_max_iters_target)
    app_linear.visualize_results(padding_factor=0.2)
    app_linear.visualizer.save_all_figures(base_filename=f"linear_T{int(initial_temp_param)}_rate{linear_cooling_rate}")

    plt.show()