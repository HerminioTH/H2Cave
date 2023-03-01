from fenics import *
import os
import sys
import numpy as np
import shutil
import json

sys.path.append(os.path.join("..", "..", "libs"))
from GridHandler import GridHandler
from ResultsHandler import AverageSaver, VtkSaver
from TimeHandler import TimeHandler
from SimulationHandler import *
from Models import *

def pressure(t):
	p_int = 13*MPa
	period = 2*np.pi/(2*day)
	dP = 4*MPa
	return (p_int + dP*np.sin(period*t))


def main():
	# Define settings
	time_list = np.linspace(0, 2*day, 30)
	pressures = pressure(time_list)

	settings = {
		"Paths" : {
			"Output": "output/case_1",
			"Grid": "../../grids/cavern_0",
		},
		"Time" : {
			"timeList": time_list,
			"timeStep": 10*hour,
			"finalTime": 800*hour,
			"theta": 0.5,
		},
		"Viscoelastic" : {
			"active": True,
			"E0": 32*GPa,
			"nu0": 0.3,
			"E1": 2*GPa,
			"nu1": 0.3,
			"eta": 5e14
		},
		"DislocationCreep" : {
			"active": True,
			"A": 5.2e-36,
			"n": 5.0,
			"T": 298
		},
		"BoundaryConditions" : {
			"u_x" : {
				"SIDE_X": 	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"SIDE_Y": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"OUTER": 	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"BOTTOM":	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"TOP": 		{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"WALL":		{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))}
			},
			"u_y" : {
				"SIDE_X": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"SIDE_Y": 	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"OUTER": 	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"BOTTOM":	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"TOP": 		{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"WALL":		{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))}
			},
			"u_z" : {
				"SIDE_X": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"SIDE_Y": 	{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"OUTER": 	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"BOTTOM":	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"TOP": 		{"type": "NEUMANN", 	"value": np.repeat(0.0, len(time_list))},
				"WALL":		{"type": "NEUMANN", 	"value": pressures}
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
	bc_handler = BoundaryConditionHandler(fem_handler, settings)

	# Build salt model
	salt = SaltModel_2(fem_handler, bc_handler, settings)

	# Saver
	avg_eps_tot_saver = AverageSaver(fem_handler.dx, "eps_tot", salt.model_v.eps_tot, time_handler, output_folder)
	avg_eps_v_saver = AverageSaver(fem_handler.dx, "eps_v", salt.model_v.eps_v, time_handler, output_folder)
	avg_eps_e_saver = AverageSaver(fem_handler.dx, "eps_e", salt.model_v.eps_e, time_handler, output_folder)
	avg_eps_cr_saver = AverageSaver(fem_handler.dx, "eps_cr", salt.model_c.eps_cr, time_handler, output_folder)

	vtk_u_saver = VtkSaver("displacement", salt.u, time_handler, output_folder)
	vtk_stress_saver = VtkSaver("stress", salt.model_v.stress, time_handler, output_folder)
	vtk_eps_tot_saver = VtkSaver("eps_tot", salt.model_v.eps_tot, time_handler, output_folder)

	# Define simulator
	sim = Simulator(salt, time_handler)

	# Add savers
	sim.add_saver(avg_eps_tot_saver)
	sim.add_saver(avg_eps_v_saver)
	sim.add_saver(avg_eps_e_saver)
	sim.add_saver(avg_eps_cr_saver)
	sim.add_saver(vtk_u_saver)
	sim.add_saver(vtk_eps_tot_saver)

	# Run simulation
	sim.run()

	# Copy .msh mesh to output_folder
	source = os.path.join(grid_folder, geometry_name+".msh")
	destination = output_folder
	shutil.copy(source, destination)
	shutil.copy(__file__, os.path.join(destination, "copy.py"))








if __name__ == "__main__":
	main()