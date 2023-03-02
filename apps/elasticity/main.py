from fenics import *
import os
import sys
import numpy as np
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandler
from Events import VtkSaver, AverageSaver, ScreenOutput, TimeLevelCounter
from Controllers import TimeController, IterationController, ErrorController
from Time import TimeHandler
from FiniteElements import FemHandler
from BoundaryConditions import MechanicsBoundaryConditions
from Simulators import Simulator
from Models import ElasticModel
from Utils import *

def write_settings(settings):
	# Define time levels
	n_steps = 50
	t_f = 800*hour
	settings["Time"]["timeList"] = list(np.linspace(0, t_f, n_steps))

	# Define boundary conditions
	for u_i in settings["BoundaryConditions"].keys():
		for boundary_name in settings["BoundaryConditions"][u_i]:
			settings["BoundaryConditions"][u_i][boundary_name]["value"] = list(np.repeat(0.0, n_steps))

	settings["BoundaryConditions"]["u_z"]["TOP"]["value"] = list(np.linspace(-12*MPa, -15*MPa, n_steps))

	# Dump to file
	save_json(settings, "settings.json")

def main():
	# Read settings
	settings = read_json("settings.json")

	# Write settings
	write_settings(settings)

	# Define folders
	output_folder = os.path.join(*settings["Paths"]["Output"].split("/"))
	grid_folder = os.path.join(*settings["Paths"]["Grid"].split("/"))

	# Load grid
	geometry_name = "geom"
	grid = GridHandler(geometry_name, grid_folder)

	# Define time handler
	time_handler = TimeHandler(settings["Time"])

	# Define finite element handler (function spaces, normal vectors, etc)
	fem_handler = FemHandler(grid)

	# Define boundary condition handler
	bc_handler = MechanicsBoundaryConditions(fem_handler, settings)

	# Define model
	model = ElasticModel(fem_handler, bc_handler, settings)

	# Controllers
	time_controller = TimeController("Time (s)", time_handler)

	# Events
	avg_eps_tot_saver = AverageSaver(fem_handler.dx(), "eps_tot", model.elastic_element.eps_tot, time_handler, output_folder)
	avg_eps_e_saver = AverageSaver(fem_handler.dx(), "eps_e", model.elastic_element.eps_e, time_handler, output_folder)

	vtk_u_saver = VtkSaver("displacement", model.u, time_handler, output_folder)
	vtk_stress_saver = VtkSaver("stress", model.elastic_element.stress, time_handler, output_folder)

	time_counter = TimeLevelCounter(time_handler)

	screen_monitor = ScreenOutput()
	screen_monitor.add_controller(time_counter, width=10, align="center")
	screen_monitor.add_controller(time_controller, width=30, align="center")

	# Define simulator
	sim = Simulator(time_handler)

	# Add models
	sim.add_model(model)

	# Add events
	sim.add_event(avg_eps_tot_saver)
	sim.add_event(avg_eps_e_saver)
	sim.add_event(vtk_u_saver)
	sim.add_event(vtk_stress_saver)
	sim.add_event(time_counter)
	sim.add_event(screen_monitor)

	# Add controllers
	sim.add_controller(time_controller)

	# Run simulator
	sim.run()

if __name__ == "__main__":
	main()