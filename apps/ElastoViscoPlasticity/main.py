import os
import sys
import numpy as np
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandler
from Events import VtkSaver, AverageSaver, ScreenOutput, TimeLevelCounter, TimeCounter
from Controllers import TimeController, IterationController, ErrorController
from Time import TimeHandler
from FiniteElements import FemHandler
from BoundaryConditions import MechanicsBoundaryConditions
from Simulators import Simulator
from Models import ElastoViscoplasticModel
from Utils import *

# ================ Burgers model ================== #
#  \|                     ____                      #
#  \|               ___  | σY |____                 #
#  \|     E0,𝜈     |   \ +————+    |                #
#  \|__  /\  /\  __|    ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅    |———--🢂 σ       #
#  \|  \/  \/  \/  |   _________   |                #
#  \|              |_____|      |__|                #
#  \|                    |  η1  |                   #
#                      ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅                     #
#   |———— Ɛ_e ———_—|———— Ɛ_ie —————|                #
#   |——————————— Ɛ_tot ————————————|                #
# ================================================= #


def write_settings(settings):
	# Define time levels
	n_steps = 3
	t_f = 20*hour
	settings["Time"]["timeList"] = list(np.linspace(0, t_f, n_steps))

	# Define boundary conditions
	for u_i in settings["BoundaryConditions"].keys():
		for boundary_name in settings["BoundaryConditions"][u_i]:
			settings["BoundaryConditions"][u_i][boundary_name]["value"] = list(np.repeat(0.0, n_steps))

	settings["BoundaryConditions"]["u_z"]["TOP"]["value"] = list(np.repeat(35*MPa, n_steps))
	settings["BoundaryConditions"]["u_x"]["OUTSIDE"]["value"] = list(np.repeat(0*MPa, n_steps))

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
	grid = GridHandler("geom", grid_folder)

	# Define time handler
	time_handler = TimeHandler(settings["Time"])

	# Define finite element handler (function spaces, normal vectors, etc)
	fem_handler = FemHandler(grid)

	# Define boundary condition handler
	bc_handler = MechanicsBoundaryConditions(fem_handler, settings)

	# Define model
	model = ElastoViscoplasticModel(fem_handler, bc_handler, settings)

	# Controllers
	iteration_controller = IterationController("Iterations", max_ite=10)
	error_controller = ErrorController("Error", model, tol=1e-5)

	# Events
	avg_eps_tot_saver = AverageSaver(fem_handler.dx(), "eps_tot", model.elastic_element.eps_tot, time_handler, output_folder)
	avg_eps_e_saver = AverageSaver(fem_handler.dx(), "eps_e", model.elastic_element.eps_e, time_handler, output_folder)
	avg_eps_ie_saver = AverageSaver(fem_handler.dx(), "eps_ie", model.viscoplastic_element.eps_ie, time_handler, output_folder)

	vtk_u_saver = VtkSaver("displacement", model.u, time_handler, output_folder)
	vtk_stress_saver = VtkSaver("stress", model.elastic_element.stress, time_handler, output_folder)
	vtk_eps_e_saver = VtkSaver("eps_e", model.elastic_element.eps_e, time_handler, output_folder)
	vtk_eps_ie_saver = VtkSaver("eps_ie", model.viscoplastic_element.eps_ie, time_handler, output_folder)
	vtk_alpha_saver = VtkSaver("alpha", model.viscoplastic_element.alpha, time_handler, output_folder)
	vtk_I1_saver = VtkSaver("I1", model.viscoplastic_element.I1, time_handler, output_folder)
	vtk_J2_saver = VtkSaver("J2", model.viscoplastic_element.J2, time_handler, output_folder)

	time_level_counter = TimeLevelCounter(time_handler)
	time_counter = TimeCounter(time_handler)

	screen_monitor = ScreenOutput()
	screen_monitor.add_controller(time_level_counter, width=10, align="center")
	screen_monitor.add_controller(time_counter, width=20, align="center")
	screen_monitor.add_controller(iteration_controller, width=10, align="center")
	screen_monitor.add_controller(error_controller, width=30, align="center")

	# Define simulator
	sim = Simulator(time_handler)

	# Add models
	sim.add_model(model)

	# Add events
	sim.add_event(avg_eps_tot_saver)
	sim.add_event(avg_eps_e_saver)
	sim.add_event(avg_eps_ie_saver)
	sim.add_event(vtk_u_saver)
	sim.add_event(vtk_stress_saver)
	sim.add_event(vtk_eps_e_saver)
	sim.add_event(vtk_eps_ie_saver)
	sim.add_event(vtk_alpha_saver)
	sim.add_event(vtk_I1_saver)
	sim.add_event(vtk_J2_saver)
	sim.add_event(time_level_counter)
	sim.add_event(time_counter)
	sim.add_event(screen_monitor)

	# Add controllers
	sim.add_controller(iteration_controller)
	sim.add_controller(error_controller)

	# Run simulator
	sim.run()

if __name__ == "__main__":
	main()