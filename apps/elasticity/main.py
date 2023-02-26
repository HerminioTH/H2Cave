from fenics import *
import os
import sys
import numpy as np
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandler
from Events import VtkSaver, AverageSaver, ScreenOutput
from Controllers import TimeController
from Time import TimeHandler
from FiniteElements import FemHandler
from BoundaryConditions import MechanicsBoundaryConditions
from Simulators import Simulator_3
from ModelCompositions import ElasticModel
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
	model = ElasticModel(fem_handler, bc_handler, settings)

	# Controllers
	time_controller = TimeController("Time", time_handler)

	# Events
	avg_eps_tot_saver = AverageSaver(fem_handler.dx(), "eps_tot", model.elastic_element.eps_tot, time_handler, output_folder)
	avg_eps_e_saver = AverageSaver(fem_handler.dx(), "eps_e", model.elastic_element.eps_e, time_handler, output_folder)

	vtk_u_saver = VtkSaver("displacement", model.u, time_handler, output_folder)
	vtk_stress_saver = VtkSaver("stress", model.elastic_element.stress, time_handler, output_folder)

	screen_monitor = ScreenOutput()
	screen_monitor.add_controller(time_controller, width=10, align="center")

	# Define simulator
	sim = Simulator_3(time_handler)

	# Add models
	sim.add_model(model)

	# Add events
	sim.add_event(avg_eps_tot_saver)
	sim.add_event(avg_eps_e_saver)
	sim.add_event(vtk_u_saver)
	sim.add_event(vtk_stress_saver)
	sim.add_event(screen_monitor)

	# Add controllers
	sim.add_controller(time_controller)

	# Run simulator
	sim.run()

if __name__ == "__main__":
	main()