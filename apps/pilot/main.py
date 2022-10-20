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
from Models import *


sec = 1.
minute = 60*sec
hour = 60*minute
day = 24*hour
kPa = 1e3
MPa = 1e6
GPa = 1e9


# class Simulator():
# 	def __init__(self)





def main():
	# Define settings
	time_list = np.linspace(0, 80*hour, 10)
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
		"Viscoelastic" : {
			"active": True,
			"E0": 2*GPa,
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

	# Define time handler
	time_handler = TimeHandler(settings["Time"])

	# Define folders
	output_folder = os.path.join(*settings["Paths"]["Output"].split("/"))
	grid_folder = os.path.join(*settings["Paths"]["Grid"].split("/"))

	# Load grid
	geometry_name = "geom"
	grid = GridHandler(geometry_name, grid_folder)

	# Build salt model
	salt = SaltModel(grid, settings)

	# Saver
	avg_saver = AverageSaver(salt.dx, salt.model_v.eps_tot, time_handler, output_folder)
	vtk_saver = VtkSaver("displacement", salt.u, time_handler, output_folder)



	# Update boundary conditions
	salt.update_BCs(time_handler)

	# Solve instantaneoyus elastic response
	salt.solve_elastic_model(time_handler)

	# Compute total strain
	salt.compute_total_strain()

	# Compute elastic strin
	salt.compute_elastic_strain()

	# Compute stress
	salt.compute_stress()

	# Compute viscous strain
	salt.update_matrices(time_handler)
	salt.compute_viscous_strain()
	salt.model_v.update()
	salt.assemble_matrix()

	# Save results
	avg_saver.record()
	vtk_saver.record()

	# Time marching
	while not time_handler.is_final_time_reached():

		# Update time
		time_handler.advance_time()
		print()
		print(time_handler.time/hour)

		# Update constitutive matrices
		salt.update_matrices(time_handler)

		# Assemble stiffness matrix
		salt.assemble_matrix()

		# Compute creep
		salt.compute_creep(time_handler)

		# Update Neumann BC
		salt.update_BCs(time_handler)

		# Iteration settings
		ite = 0
		tol = 1e-9
		error = 2*tol
		error_old = error

		while error > tol:
			# Solve mechanical problem
			salt.solve_mechanics()

			# Compute error
			error = salt.compute_error()
			print(ite, error)

			# Compute total strain
			salt.compute_total_strain()

			# Compute elastic strin
			salt.compute_elastic_strain()

			# Compute stress
			salt.compute_stress()

			# Compute creep
			salt.compute_creep(time_handler)

			# Increase iteration
			ite += 1

		# Compute viscous strain
		salt.compute_viscous_strain()

		# Update old strins
		salt.update_old_strains()

		# Save results
		avg_saver.record()
		vtk_saver.record()

	# Write results
	avg_saver.save()
	vtk_saver.save()







if __name__ == "__main__":
	main()