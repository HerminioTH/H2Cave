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
from Models import BurgersModel
from Elements import DislocationCreep, PressureSolutionCreep
from Utils import *

# =========================== Creep model - 1 ============================ #
#  \|                      E1                                              #
#  \|               ___  /\  /\  __                                        #
#  \|     E0,𝜈     |   \/  \/  \/  |   _________    _________              #
#  \|__  /\  /\  __|               |_____|      |_____|      |———--🢂 σ    #
#  \|  \/  \/  \/  |   _________   |     |  η2  |     |  η3  |             #
#  \|              |_____|      |__|   ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅     ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅               #
#  \|                    |  η1  |                                          #
#                      ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅                                            #
#   |——— Ɛ_e ————|—————— Ɛ_v ——————|——— Ɛ_d ————|——— Ɛ_p ————|             #
#   |————————————————————————— Ɛ_tot ————————————————————————|             #
#                                                                          #
#   Ɛ_e   - elastic strain                                                 #
#   Ɛ_v   - viscoelastic strain                                            #
#   Ɛ_d   - dislocation creep strain                                       #
#   Ɛ_p   - pressure solution creep strain                                 #
#   Ɛ_tot - total strain                                                   #
# ======================================================================== #

def write_settings(settings):
	# Define time levels
	n_steps = 25
	t_f = 40*hour
	settings["Time"]["timeList"] = list(np.linspace(0, t_f, n_steps))

	# Define boundary conditions
	for u_i in settings["BoundaryConditions"].keys():
		for boundary_name in settings["BoundaryConditions"][u_i]:
			settings["BoundaryConditions"][u_i][boundary_name]["value"] = list(np.repeat(0.0, n_steps))

	settings["BoundaryConditions"]["u_z"]["TOP"]["value"] = list(np.repeat(16*MPa, n_steps))

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

	# Build Burgers model
	model = BurgersModel(fem_handler, bc_handler, settings)
	model.add_inelastic_element(DislocationCreep(fem_handler, settings, element_name="DislocationCreep"))
	model.add_inelastic_element(PressureSolutionCreep(fem_handler, settings, element_name="PressureSolutionCreep"))

	# Controllers
	iteration_controller = IterationController("Iterations", max_ite=20)
	error_controller = ErrorController("Error", model, tol=1e-8)

	# Events
	avg_eps_tot_saver = AverageSaver(fem_handler.dx(), "eps_tot", model.viscoelastic_element.eps_tot, time_handler, output_folder)
	avg_eps_e_saver = AverageSaver(fem_handler.dx(), "eps_e", model.viscoelastic_element.eps_e, time_handler, output_folder)
	avg_eps_ve_saver = AverageSaver(fem_handler.dx(), "eps_ve", model.viscoelastic_element.eps_v, time_handler, output_folder)
	avg_eps_d_saver = AverageSaver(fem_handler.dx(), "eps_d", model.inelastic_elements[0].eps_ie, time_handler, output_folder)
	avg_eps_p_saver = AverageSaver(fem_handler.dx(), "eps_p", model.inelastic_elements[1].eps_ie, time_handler, output_folder)
	avg_stress_saver = AverageSaver(fem_handler.dx(), "stress", model.viscoelastic_element.stress, time_handler, output_folder)

	vtk_u_saver = VtkSaver("displacement", model.u, time_handler, output_folder)

	time_level_counter = TimeLevelCounter(time_handler)
	time_counter = TimeCounter(time_handler, "Time (h)", "hours")

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
	sim.add_event(avg_eps_ve_saver)
	sim.add_event(avg_eps_d_saver)
	sim.add_event(avg_eps_p_saver)
	sim.add_event(avg_stress_saver)
	sim.add_event(vtk_u_saver)
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