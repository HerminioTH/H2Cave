import os
import sys
import numpy as np
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandler
from Events import VtkSaver, AverageSaver, AverageScalerSaver, ScreenOutput, TimeLevelCounter, TimeCounter
from Controllers import TimeController, IterationController, ErrorController
from Time import TimeHandler
from FiniteElements import FemHandler
from BoundaryConditions import MechanicsBoundaryConditions
from Simulators import Simulator
from Models import BurgersModel
from Elements import DislocationCreep, Damage
from Utils import *
import time

# ========================== Damage model - 1 ============================ #
#  \|                      E1                                              #
#  \|               ___  /\  /\  __                                        #
#  \|     E0,𝜈     |   \/  \/  \/  |   _________    _________              #
#  \|__  /\  /\  __|               |_____|      |_____|      |———--🢂 σ    #
#  \|  \/  \/  \/  |   _________   |     |  η2  |     |  η3  |             #
#  \|              |_____|      |__|   ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅     ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅               #
#  \|                    |  η1  |                                          #
#                      ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅                                            #
#   |——— Ɛ_e ————|—————— Ɛ_v ——————|——— Ɛ_cr ———|——— Ɛ_dm ———|             #
#   |————————————————————————— Ɛ_tot ————————————————————————|             #
#                                                                          #
#   Ɛ_e   - elastic strain                                                 #
#   Ɛ_v   - viscoelastic strain                                            #
#   Ɛ_cr  - dislocation creep strain                                       #
#   Ɛ_dm  - strain associated to damage evolution (tertiary creep)         #
#   Ɛ_tot - total strain                                                   #
# ======================================================================== #

def write_settings(settings):
	# Define time levels
	n_steps = 400
	t_f = 2716*hour
	settings["Time"]["timeList"] = list(np.linspace(0, t_f, n_steps))

	# Define boundary conditions
	for u_i in settings["BoundaryConditions"].keys():
		for boundary_name in settings["BoundaryConditions"][u_i]:
			settings["BoundaryConditions"][u_i][boundary_name]["value"] = list(np.repeat(0.0, n_steps))

	settings["BoundaryConditions"]["u_z"]["TOP"]["value"] = list(np.repeat(-18.7*MPa, n_steps))

	# Dump to file
	save_json(settings, "settings.json")

def main():
	# Read settings
	settings = read_json("settings.json")

	# Write settings
	write_settings(settings)

	# Choose damage
	use_damage = False

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
	if use_damage:
		model.add_inelastic_element(Damage(fem_handler, settings, element_name="Damage"))

	# Controllers
	iteration_controller = IterationController("Iterations", max_ite=10)
	error_controller = ErrorController("Error", model, tol=1e-5)

	# Events
	avg_eps_tot_saver = AverageSaver(fem_handler.dx(), "eps_tot", model.viscoelastic_element.eps_tot, time_handler, output_folder)

	vtk_u_saver = VtkSaver("displacement", model.u, time_handler, output_folder)
	vtk_stress_saver = VtkSaver("stress", model.viscoelastic_element.stress, time_handler, output_folder)
	vtk_eps_tot_saver = VtkSaver("eps_tot", model.viscoelastic_element.eps_tot, time_handler, output_folder)
	if use_damage:
		vtk_damage_saver = VtkSaver("damage", model.inelastic_elements[1].D, time_handler, output_folder)
		avg_damage_saver = AverageScalerSaver(fem_handler.dx(), "damage", model.inelastic_elements[1].D, time_handler, output_folder)

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
	sim.add_event(vtk_u_saver)
	sim.add_event(vtk_stress_saver)
	sim.add_event(vtk_eps_tot_saver)
	if use_damage:
		sim.add_event(vtk_damage_saver)
		sim.add_event(avg_damage_saver)
	sim.add_event(time_level_counter)
	sim.add_event(time_counter)
	sim.add_event(screen_monitor)

	# Add controllers
	sim.add_controller(iteration_controller)
	sim.add_controller(error_controller)

	# Run simulator
	start = time.time()
	sim.run()
	end = time.time()
	print("Elapsed time: %.2f"%((end-start)/day))

if __name__ == "__main__":
	main()