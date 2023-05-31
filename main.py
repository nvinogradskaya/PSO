import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QInputDialog


def plot_function():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-2, 3, 100)
    y = np.linspace(-2, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X - 2) ** 4 + (X - 2 * Y) ** 2

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X2')
    ax.set_ylabel('X2')
    ax.set_zlabel('f')
    ax.view_init(elev=10, azim=105)
    ax.dist = 7
    plt.show()
# Define the objective function
def function(x, y):
    return (x - 2) ** 4 + (x - 2 * y) ** 2

# Define the Particle class
class Particle:
    def __init__(self, x_range, y_range):
        self.position = [random.uniform(*x_range), random.uniform(*y_range)]
        self.velocity = [random.uniform(*x_range) / 10, random.uniform(*y_range) / 10]
        self.best_position = self.position.copy()

    def update_position(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        r1 = random.random()
        r2 = random.random()

        self.velocity[0] = inertia_weight * self.velocity[0] + cognitive_weight * r1 * (self.best_position[0] - self.position[0]) + social_weight * r2 * (global_best_position[0] - self.position[0])
        self.velocity[1] = inertia_weight * self.velocity[1] + cognitive_weight * r1 * (self.best_position[1] - self.position[1]) + social_weight * r2 * (global_best_position[1] - self.position[1])

    def evaluate(self):
        return function(self.position[0], self.position[1])

# Define the PSO class
class PSO:
    def __init__(self, x_range, y_range, num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight):
        self.x_range = x_range
        self.y_range = y_range
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.particles = []
        self.global_best_position = []

    def initialize_particles(self):
        self.particles = [Particle(self.x_range, self.y_range) for _ in range(self.num_particles)]
        self.global_best_position = self.particles[0].position.copy()

    def run(self):
        iteration_history = []
        for _ in range(self.num_iterations):
            iteration_data = []
            for particle in self.particles:
                fitness = particle.evaluate()

                if fitness < function(particle.best_position[0], particle.best_position[1]):
                    particle.best_position = particle.position.copy()

                if fitness < function(self.global_best_position[0], self.global_best_position[1]):
                    self.global_best_position = particle.position.copy()

                particle.update_velocity(self.global_best_position, self.inertia_weight, self.cognitive_weight, self.social_weight)
                particle.update_position()

                iteration_data.append((particle.position[0], particle.position[1], fitness))

            iteration_history.append(iteration_data)

        return iteration_history, self.global_best_position, function(self.global_best_position[0], self.global_best_position[1])

# Define the main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Роевой алгоритм частиц")
        self.setGeometry(200, 200, 1250, 500)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.create_input_widgets()
        self.create_table_widget()
        self.create_result_label()
        self.create_buttons()

    def create_input_widgets(self):
        self.input_layout = QVBoxLayout()

        self.num_particles_label = QLabel("Количество частиц:")
        self.num_particles_input = QLineEdit()
        self.input_layout.addWidget(self.num_particles_label)
        self.input_layout.addWidget(self.num_particles_input)

        self.num_iterations_label = QLabel("Количество итераций:")
        self.num_iterations_input = QLineEdit()
        self.input_layout.addWidget(self.num_iterations_label)
        self.input_layout.addWidget(self.num_iterations_input)

        self.inertia_weight_label = QLabel("Инерционный коэфф-т:")
        self.inertia_weight_input = QLineEdit()
        self.input_layout.addWidget(self.inertia_weight_label)
        self.input_layout.addWidget(self.inertia_weight_input)

        self.cognitive_weight_label = QLabel("Когнитивный коэфф-т [0,1]:")
        self.cognitive_weight_input = QLineEdit()
        self.input_layout.addWidget(self.cognitive_weight_label)
        self.input_layout.addWidget(self.cognitive_weight_input)

        self.social_weight_label = QLabel("Социальный коэфф-т [0,1]:")
        self.social_weight_input = QLineEdit()
        self.input_layout.addWidget(self.social_weight_label)
        self.input_layout.addWidget(self.social_weight_input)

        self.x_range_label = QLabel("Диапазон х1:")
        self.x_range_min_input = QLineEdit()
        self.x_range_max_input = QLineEdit()
        self.input_layout.addWidget(self.x_range_label)
        self.input_layout.addWidget(self.x_range_min_input)
        self.input_layout.addWidget(self.x_range_max_input)

        self.y_range_label = QLabel("Диапазон х2:")
        self.y_range_min_input = QLineEdit()
        self.y_range_max_input = QLineEdit()
        self.input_layout.addWidget(self.y_range_label)
        self.input_layout.addWidget(self.y_range_min_input)
        self.input_layout.addWidget(self.y_range_max_input)

        self.layout.addLayout(self.input_layout)

    def create_table_widget(self):
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Частица х1", "Частица х2", "Значение f"])
        self.table_widget.setColumnWidth(0, 150)  # Set width of the first column
        self.table_widget.setColumnWidth(1, 150)  # Set width of the second column
        self.table_widget.setColumnWidth(2, 150)  # Set width of the third column
        self.layout.addWidget(self.table_widget, 2)

    def create_result_label(self):
        self.result_label = QLabel()
        self.layout.addWidget(self.result_label, 2)

    def create_buttons(self):
        self.buttons_layout = QVBoxLayout()

        self.start_button = QPushButton("Старт")
        self.start_button.clicked.connect(self.run_algorithm)
        self.buttons_layout.addWidget(self.start_button)

        self.show_iteration_button = QPushButton("Показать итерацию")
        self.show_iteration_button.clicked.connect(self.show_iteration)
        self.show_iteration_button.setEnabled(False)
        self.buttons_layout.addWidget(self.show_iteration_button)

        self.layout.addLayout(self.buttons_layout)

    def run_algorithm(self):
        num_particles = int(self.num_particles_input.text())
        num_iterations = int(self.num_iterations_input.text())
        inertia_weight = float(self.inertia_weight_input.text())
        cognitive_weight = float(self.cognitive_weight_input.text())
        social_weight = float(self.social_weight_input.text())
        x_range = (float(self.x_range_min_input.text()), float(self.x_range_max_input.text()))
        y_range = (float(self.y_range_min_input.text()), float(self.y_range_max_input.text()))

        pso = PSO(x_range, y_range, num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight)
        pso.initialize_particles()
        iteration_history, best_position, best_value = pso.run()

        self.iteration_history = iteration_history
        self.best_position = best_position
        self.best_value = best_value

        self.show_iteration_button.setEnabled(True)

        self.populate_table(iteration_history[-1])
        self.show_result()

    def populate_table(self, iteration_data):
        self.table_widget.clearContents()
        self.table_widget.setRowCount(len(iteration_data))

        for row, data in enumerate(iteration_data):
            x_item = QTableWidgetItem(format(data[0], ".5f"))
            y_item = QTableWidgetItem(format(data[1], ".5f"))
            value_item = QTableWidgetItem(format(data[2], ".5f"))

            self.table_widget.setItem(row, 0, x_item)
            self.table_widget.setItem(row, 1, y_item)
            self.table_widget.setItem(row, 2, value_item)

    def show_iteration(self):
        iteration, ok = QInputDialog.getInt(self, "Выбор итерации", "Номер итерации:", 1, 1, len(self.iteration_history))
        if ok:
            iteration_data = self.iteration_history[iteration - 1]
            self.populate_table(iteration_data)

    def show_result(self):
        result_text = f"Лучшее решение: x1 = {self.best_position[0]}, х2 = {self.best_position[1]}\n"
        result_text += f"Минимум функции: {self.best_value}"
        self.result_label.setText(result_text)

if __name__ == "__main__":
    plot_function()
    # Create the application
    app = QApplication(sys.argv)
    # Create the main window
    window = MainWindow()
    window.show()
    # Execute the application
    sys.exit(app.exec())
