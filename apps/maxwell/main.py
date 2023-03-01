from fenics import *
import os
import sys
import numpy as np
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandler
from Events import VtkSaver, AverageSaver, ScreenOutput
from Controllers import TimeController, IterationController, ErrorController
from Time import TimeHandler
from FiniteElements import FemHandler
from BoundaryConditions import MechanicsBoundaryConditions
from Simulators import Simulator
from Models import MaxwellModel
from Utils import *

def main():
	# Define settings
	time_list = np.linspace(0, 800*hour, 50)
	settings = {
		"Paths" : {
			"Output": "output/case_1",
			"Grid": "../../grids/quarter_cylinder_0",
		},
		"Time" : {
			"timeList": time_list,
			"timeStep": 10*hour,
			"finalTime": 800*hour,
			"theta": 0.5,
		},
		"Elastic" : {
			"E": 2*GPa,
			"nu": 0.3
		},
		"Dashpot" : {
			"eta": 2e15
		},
		"BoundaryConditions" : {
			"u_x" : {
				"SIDE_X": 	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"SIDE_Y": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"OUTSIDE": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"BOTTOM":	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"TOP": 		{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))}
			},
			"u_y" : {
				"SIDE_X": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"SIDE_Y": 	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"OUTSIDE": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"BOTTOM":	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"TOP": 		{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))}
			},
			"u_z" : {
				"SIDE_X": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"SIDE_Y": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"OUTSIDE": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"BOTTOM":	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"TOP": 		{"type": "NEUMANN", 	"value": np.repeat(-12*MPa, len(time_list))}
				# "TOP": 		{"type": "NEUMANN", 	"value": np.linspace(-12*MPa, -15*MPa, len(time_list))}
			}
		}
	}

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
	model = MaxwellModel(fem_handler, bc_handler, settings)

	# Controllers
	time_controller = TimeController("Time", time_handler)
	iteration_controller = IterationController("Iteration", max_ite=10)
	# error_controller = ErrorController("Error", model, tol=1e-8)
	error_controller = ErrorController("||u_k - u_k-1||/||u_k||", model, tol=1e-5)

	# Events
	avg_eps_tot_saver = AverageSaver(fem_handler.dx(), "eps_tot", model.elastic_element.eps_tot, time_handler, output_folder)
	avg_eps_e_saver = AverageSaver(fem_handler.dx(), "eps_e", model.elastic_element.eps_e, time_handler, output_folder)
	avg_eps_ie_saver = AverageSaver(fem_handler.dx(), "eps_ie", model.dashpot_element.eps_ie, time_handler, output_folder)

	vtk_u_saver = VtkSaver("displacement", model.u, time_handler, output_folder)
	vtk_stress_saver = VtkSaver("stress", model.elastic_element.stress, time_handler, output_folder)
	vtk_eps_e_saver = VtkSaver("eps_e", model.elastic_element.eps_e, time_handler, output_folder)
	vtk_eps_ie_saver = VtkSaver("eps_ie", model.dashpot_element.eps_ie, time_handler, output_folder)

	screen_monitor = ScreenOutput()
	screen_monitor.add_controller(time_controller, width=10, align="center")
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
	sim.add_event(screen_monitor)

	# Add controllers
	sim.add_controller(iteration_controller)
	sim.add_controller(error_controller)

	# Run simulator
	sim.run()

if __name__ == "__main__":
	main()