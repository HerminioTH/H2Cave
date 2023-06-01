import os
import sys
import numpy as np
import time
import shutil
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandler
from Events import VtkSaver, AverageSaver, AverageScalerSaver, ScreenOutput, TimeLevelCounter, TimeCounter
from Controllers import TimeController, IterationController, ErrorController
from Time import TimeHandler
from FiniteElements import FemHandler
from BoundaryConditions import MechanicsBoundaryConditions
from Simulators import Simulator
from Models import MaxwellModel_2, BurgersModel, ElasticModel, ViscoelasticModel
from Elements import DashpotElement, DislocationCreep, PressureSolutionCreep, Damage
from Utils import *

# =============================== Complete model ====================================== #
#  \|                      E1,𝜈                                                         #
#  \|               ___  /\  /\  __                                                     #
#  \|     E0,𝜈     |   \/  \/  \/  |   _________    _________    _________              #
#  \|__  /\  /\  __|               |_____|      |_____|      |_____|      |———--🢂 σ    #
#  \|  \/  \/  \/  |   _________   |     |  η2  |     |  η3  |     |  η3  |             #
#  \|              |_____|      |__|   ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅     ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅     ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅               #
#  \|                    |  η1  |                                                       #
#                      ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅                                                         #
#   |——— Ɛ_e ————|—————— Ɛ_v ——————|——— Ɛ_cr ———|——— Ɛ_ps ———|——— Ɛ_d ————|             #
#   |—————————————————————————————— Ɛ_tot ————————————————————————————————|             #
#                                                                                       #
#   Ɛ_e   - elastic strain                                                              #
#   Ɛ_v   - viscoelastic strain                                                         #
#   Ɛ_cr  - dislocation creep strain                                                    #
#   Ɛ_ps  - pressure solution creep strain                                              #
#   Ɛ_d   - damage model for teriary creep strain                                       #
#   Ɛ_tot - total strain                                                                #
# ===================================================================================== #


def write_settings(settings):
	# Define time levels
	n_steps = 20
	t_f = 24*hour
	settings["Time"]["timeList"] = list(np.linspace(0, t_f, n_steps))

	# Define boundary conditions
	for u_i in settings["BoundaryConditions"].keys():
		for boundary_name in settings["BoundaryConditions"][u_i]:
			settings["BoundaryConditions"][u_i][boundary_name]["value"] = list(np.repeat(0.0, n_steps))

	settings["BoundaryConditions"]["u_z"]["TOP"]["value"] = list(np.repeat(-12*MPa, n_steps))

	# Create a name using the elements of the model
	name = settings["Model"][0]
	for elem in settings["Model"][1:]:
		name += f"_{elem}"
	settings["Paths"]["Output"] = os.path.join(*settings["Paths"]["Output"].split("/"), name)

	# # Dump to file
	# save_json(settings, "settings.json")
	return settings


def H2Cave(settings):
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

	# Build model
	inelastic_elements = settings["Model"].copy()
	if "Spring" not in settings["Model"]:
		raise Exception("Model must have a Spring element.")

	elif "KelvinVoigt" not in settings["Model"]:
		if len(settings["Model"]) == 1:
			model = ElasticModel(fem_handler, bc_handler, settings)
		else:
			model = MaxwellModel_2(fem_handler, bc_handler, settings)

	elif "KelvinVoigt" in settings["Model"]:
		if len(settings["Model"]) == 2:
			model = ViscoelasticModel(fem_handler, bc_handler, settings)
		else:
			model = BurgersModel(fem_handler, bc_handler, settings)
		inelastic_elements.remove("KelvinVoigt")
	inelastic_elements.remove("Spring")

	# Add inelastic elements to the model
	ELEMENT_DICT = {"Dashpot": DashpotElement,
					"DislocationCreep": DislocationCreep,
					"PressureSolutionCreep": PressureSolutionCreep,
					"Damage": Damage}
	for element_name in inelastic_elements:
		model.add_inelastic_element(ELEMENT_DICT[element_name](fem_handler, settings, element_name=element_name))

	# Build simulator
	sim = Simulator(time_handler)

	# Add models
	sim.add_model(model)

	# Save displacement field
	sim.add_event(VtkSaver("displacement", model.u, time_handler, output_folder))

	# Save stress field
	if settings["Elements"]["Spring"]["save_stress_vtk"] == True:
		field_name = settings["Elements"]["Spring"]["stress_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.stress, time_handler, output_folder))
	if settings["Elements"]["Spring"]["save_stress_avg"] == True:
		field_name = settings["Elements"]["Spring"]["stress_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.stress, time_handler, output_folder))

	# Save total strain field
	if settings["Elements"]["Spring"]["save_total_strain_vtk"] == True:
		field_name = settings["Elements"]["Spring"]["total_strain_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.eps_tot, time_handler, output_folder))
	if settings["Elements"]["Spring"]["save_total_strain_avg"] == True:
		field_name = settings["Elements"]["Spring"]["total_strain_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_tot, time_handler, output_folder))

	# Save strain fields
	for element_name in settings["Model"]:
		print(element_name)
		if settings["Elements"][element_name]["save_strain_vtk"] == True:
			field_name = settings["Elements"][element_name]["strain_name"]
			sim.add_event(VtkSaver(field_name, model.elastic_element.eps_e, time_handler, output_folder))
		if settings["Elements"][element_name]["save_strain_avg"] == True:
			field_name = settings["Elements"][element_name]["strain_name"]
			sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_e, time_handler, output_folder))

	# If damage model is included, save damage field
	if "Damage" in inelastic_elements:
		i_damage = inelastic_elements.index("Damage")
		name_damage = settings["Elements"]["Damage"]["damage_name"]
		vtk_damage_saver = VtkSaver(name_damage, model.inelastic_elements[i_damage].D, time_handler, output_folder)
		avg_damage_saver = AverageScalerSaver(fem_handler.dx(), name_damage, model.inelastic_elements[i_damage].D, time_handler, output_folder)
		sim.add_event(vtk_damage_saver)
		sim.add_event(avg_damage_saver)

	# Build time counters
	time_level_counter = TimeLevelCounter(time_handler)
	time_counter = TimeCounter(time_handler, "Time (h)", "hours")
	sim.add_event(time_level_counter)
	sim.add_event(time_counter)

	# Build controllers
	if type(model) == MaxwellModel_2 or type(model) == BurgersModel:
		iteration_controller = IterationController("Iterations", max_ite=20)
		error_controller = ErrorController("Error", model, tol=1e-8)
		sim.add_controller(iteration_controller)
		sim.add_controller(error_controller)

	# Build screen monitor
	screen_monitor = ScreenOutput()
	if type(model) == ElasticModel or type(model) == ViscoelasticModel:
		screen_monitor.add_controller(time_level_counter, width=10, align="center")
		screen_monitor.add_controller(time_counter, width=20, align="center")
	else:
		screen_monitor.add_controller(time_level_counter, width=10, align="center")
		screen_monitor.add_controller(time_counter, width=20, align="center")
		screen_monitor.add_controller(iteration_controller, width=10, align="center")
		screen_monitor.add_controller(error_controller, width=30, align="center")
	sim.add_event(screen_monitor)

	# Run simlation
	start = time.time()
	sim.run()
	end = time.time()
	print("Elapsed time: %.3f"%((end-start)/minute))

	# Copy .msh mesh to output_folder
	shutil.copy(os.path.join(grid_folder, "geom.msh"), output_folder)
	# shutil.copy(__file__, os.path.join(output_folder, "copy.py"))
	save_json(settings, os.path.join(output_folder, "settings.json"))


def main():
	# Read settings
	settings = read_json("settings.json")

	# Write settings
	settings = write_settings(settings)

	# Build simulation and run
	H2Cave(settings)


if __name__ == "__main__":
	main()