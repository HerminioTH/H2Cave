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
	time_list = np.linspace(0, 800*hour, 20)
	settings = {
		"Paths" : {
			"Output": "output/case_plain",
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
				"OUTSIDE": 	{"type": "NEUMANN", 	"value": np.repeat(-2*MPa, len(time_list))},
				"BOTTOM":	{"type": "DIRICHLET", 	"value": np.repeat(0.0, len(time_list))},
				"TOP": 		{"type": "NEUMANN", 	"value": np.repeat(-10*MPa, len(time_list))}
				# "TOP": 		{"type": "NEUMANN", 	"value": np.linspace(-10*MPa, -10.001*MPa, len(time_list))}
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


	dx = Measure("dx", domain=grid.mesh, subdomain_data=grid.subdomains)
	ds = Measure("ds", domain=grid.mesh, subdomain_data=grid.boundaries)
	V = VectorFunctionSpace(grid.mesh, "CG", 1)
	TS = TensorFunctionSpace(grid.mesh, "DG", 0)
	du = TrialFunction(V)
	v = TestFunction(V)
	n = dot(v, FacetNormal(grid.mesh))

	theta = settings["Time"]["theta"]

	zero_tensor = Expression((("0.0","0.0","0.0"), ("0.0","0.0","0.0"), ("0.0","0.0","0.0")), degree=0)
	stress = local_projection(zero_tensor, TS)
	eps_tot = local_projection(zero_tensor, TS)


	E0 = Constant(settings["Elastic"]["E"])
	nu0 = Constant(settings["Elastic"]["nu"])
	eta = Constant(settings["Dashpot"]["eta"])
	C0_sy = constitutive_matrix_sy(E0, nu0)
	C0 = as_matrix(Constant(np.array(C0_sy).astype(np.float64)))

	a_form = inner(sigma(C0, epsilon(du)), epsilon(v))*dx
	A = assemble(a_form)

	u = Function(V)
	u_k = Function(V)
	# u_k = interpolate(Constant((0.0, 0.0, 0.0)), V)

	eps_e = local_projection(zero_tensor, TS)
	eps_ie = local_projection(zero_tensor, TS)
	eps_ie_old = local_projection(zero_tensor, TS)
	eps_ie_rate = local_projection(zero_tensor, TS)
	eps_ie_rate_old = local_projection(zero_tensor, TS)

	avg_eps_tot_saver = AverageSaver(fem_handler.dx(), "eps_tot", eps_tot, time_handler, output_folder)
	avg_eps_e_saver = AverageSaver(fem_handler.dx(), "eps_e", eps_e, time_handler, output_folder)
	avg_eps_ie_saver = AverageSaver(fem_handler.dx(), "eps_ie", eps_ie, time_handler, output_folder)

	tol = 1e-2
	ite_max = 10

	while not time_handler.is_final_time_reached():
		time_handler.advance_time()

		bc_handler.update_BCs(time_handler)
		error = 2*tol
		ite = 0

		keep_going = True
		while keep_going:
			b_form = inner(sigma(C0, eps_ie), epsilon(v))*dx
			b = bc_handler.b + assemble(b_form)

			[bc.apply(A, b) for bc in bc_handler.bcs]
			solve(A, u.vector(), b, "cg", "ilu")

			u_new = u.vector()
			u_old = u_k.vector()

			diff = u_new - u_old
			error = np.linalg.norm(diff)/np.linalg.norm(u_new)
			ite += 1

			eps_tot.assign(local_projection(epsilon(u), TS))
			eps_e.assign(local_projection(eps_tot - eps_ie, TS))
			stress_form = voigt2stress(dot(C0, strain2voigt(eps_e)))
			stress.assign(local_projection(stress_form, TS))
			eps_ie_rate.assign(local_projection((1/eta)*stress, TS))
			eps_ie.assign(local_projection(eps_ie_old + time_handler.time_step*(theta*eps_ie_rate_old + (1 - theta)*eps_ie_rate), TS))

			u_k.assign(u)

			if ite >= 2:
				if error <= tol:
					keep_going = False
				elif ite >= ite_max:
					keep_going = False


		print(ite, error)

		eps_ie_rate_old.assign(eps_ie_rate)
		eps_ie_old.assign(eps_ie)

		avg_eps_tot_saver.execute()
		avg_eps_e_saver.execute()
		avg_eps_ie_saver.execute()

	avg_eps_tot_saver.finalize()
	avg_eps_e_saver.finalize()
	avg_eps_ie_saver.finalize()


if __name__ == "__main__":
	main()