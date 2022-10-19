from fenics import *
import os
import sys
import numpy as np
import shutil
import json

sys.path.append(os.path.join("..", "..", "libs"))
from GridHandler import GridHandler
from ResultsHandler import TensorSaver
from TimeHandler import TimeHandler
from Models import *

sec = 1.
minute = 60*sec
hour = 60*minute
day = 24*hour
kPa = 1e3
MPa = 1e6
GPa = 1e9

def axial_stress(t):
	return 12*MPa

def compute_error(u, u_k):
	return np.linalg.norm(u.vector() - u_k.vector()) / np.linalg.norm(u.vector())

class SaltModel():
	def __init__(self, grid, settings):
		self.grid = grid
		self.settings = settings
		self.initialize()

	def initialize(self):
		self.define_measures()
		self.define_function_spaces()
		self.define_trial_test_functions()
		self.define_solution_vector()
		self.define_outward_directions()
		self.define_viscoelastic_model()
		self.define_dislocation_creep_model()
		self.define_DirichletBC()

	def define_measures(self):
		self.dx = Measure("dx", domain=self.grid.mesh, subdomain_data=self.grid.subdomains)
		self.ds = Measure("ds", domain=self.grid.mesh, subdomain_data=self.grid.boundaries)

	def define_function_spaces(self):
		self.V = VectorFunctionSpace(self.grid.mesh, "CG", 1)
		self.TS = TensorFunctionSpace(self.grid.mesh, "DG", 0)

	def define_trial_test_functions(self):
		self.du = TrialFunction(self.V)
		self.v = TestFunction(self.V)

	def define_outward_directions(self):
		norm = FacetNormal(self.grid.mesh)
		self.v_n = dot(self.v, norm)

	def define_viscoelastic_model(self):
		self.model_v = ViscoElasticModel(self.settings["Viscoelastic"], self.settings["Time"]["theta"], self.du, self.v, self.dx, self.TS)

	def define_dislocation_creep_model(self):
		self.model_c = CreepDislocation(self.settings["DislocationCreep"], self.settings["Time"]["theta"], self.model_v.C0, self.du, self.v, self.dx, self.TS)

	def define_solution_vector(self):
		self.u = Function(self.V)
		self.u_k = Function(self.V)
		self.u.rename("Displacement", "m")
		self.u_k.rename("Displacement", "m")

	def define_DirichletBC(self):
		self.bcs = []
		self.bcs.append(DirichletBC(self.V.sub(2), Constant(0.0), self.grid.boundaries, self.grid.dolfin_tags[self.grid.boundary_dim]["BOTTOM"]))
		self.bcs.append(DirichletBC(self.V.sub(1), Constant(0.0), self.grid.boundaries, self.grid.dolfin_tags[self.grid.boundary_dim]["SIDE_Y"]))
		self.bcs.append(DirichletBC(self.V.sub(0), Constant(0.0), self.grid.boundaries, self.grid.dolfin_tags[self.grid.boundary_dim]["SIDE_X"]))

	def update_Neumann_BC(self, time_handler):
		L_bc = -axial_stress(time_handler.time)*self.v_n*ds(self.grid.dolfin_tags[self.grid.boundary_dim]["TOP"])
		self.b_bc = assemble(L_bc)

	def solve_elastic_model(self, time_handler):
		# Initialize matrix
		self.model_v.build_A_elastic()
		A = self.model_v.A_elastic

		# Apply Neumann boundary conditions
		self.update_Neumann_BC(time_handler)
		b = self.b_bc
		# L_bc = -23*MPa*self.v_n*ds(self.grid.dolfin_tags[self.grid.boundary_dim]["TOP"])
		# b = assemble(L_bc)

		# Solve instantaneous elastic problem
		[bc.apply(A, b) for bc in self.bcs]
		solve(A, self.u.vector(), b, "cg", "ilu")

		print(np.linalg.norm(self.u.vector()))

	def solve_mechanics(self):
		# Initialize rhs
		b = 0

		# Build creep rhs
		self.model_c.build_b()
		b += self.model_c.b

		# Build viscoelastic rhs
		self.model_v.build_b(self.model_c.eps_cr, self.model_c.eps_cr_old)
		b += self.model_v.b

		# Boundary condition rhs
		b += self.b_bc

		# Apply Dirichlet boundary conditions
		[bc.apply(self.model_v.A, b) for bc in self.bcs]

		# Solve linear system
		solve(self.model_v.A, self.u.vector(), b, "cg", "ilu")

	def assemble_matrix(self):
		# Assemble matrix
		self.model_v.build_A()

	def compute_total_strain(self):
		self.model_v.compute_total_strain(self.u)

	def compute_elastic_strain(self):
		self.model_v.compute_elastic_strain(self.model_c.eps_cr)

	def compute_stress(self):
		self.model_v.compute_stress()

	def compute_creep(self, time_handler):
		self.model_c.compute_creep_strain(self.model_v.stress, time_handler.time_step)

	def compute_viscous_strain(self):
		self.model_v.compute_viscous_strain(self.model_c.eps_cr, self.model_c.eps_cr_old)

	def update_old_strains(self):
		self.model_v.update()
		self.model_c.update()

	def update_matrices(self, time_handler):
		self.model_v.compute_matrices(time_handler.time_step)

	def compute_error(self):
		error = np.linalg.norm(self.u.vector() - self.u_k.vector()) / np.linalg.norm(self.u.vector())
		self.u_k.assign(self.u)
		return error



def main():
	# Define settings
	settings = {
		"Paths" : {
			"Output": "output/case_1",
			"Grid": "../../grids/quarter_cylinder_0",
		},
		"Time" : {
			"timeList": None,
			"timeStep": 10*hour,
			"finalTime": 30*hour,
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
	saver_eps_tot = TensorSaver("eps_tot", salt.dx)

	# Solve instantaneoyus elastic response
	salt.solve_elastic_model(time_handler)

	# Compute total strain
	salt.compute_total_strain()

	# Compute elastic strin
	salt.compute_elastic_strain()

	# Compute stress
	salt.compute_stress()

	# Save results
	saver_eps_tot.record_average(salt.model_v.eps_tot, time_handler.time)

	# # Time marching
	# while not time_handler.is_final_time_reached():

	# 	# Update time
	# 	time_handler.advance_time()
	# 	print()
	# 	print(time_handler.time/hour)

	# 	# Update constitutive matrices
	# 	salt.update_matrices(time_handler)

	# 	# Assemble stiffness matrix
	# 	salt.assemble_matrix()

	# 	# Compute creep
	# 	salt.compute_creep(time_handler)

	# 	# Update Neumann BC
	# 	salt.update_Neumann_BC(time_handler)

	# 	# Iteration settings
	# 	ite = 0
	# 	tol = 1e-9
	# 	error = 2*tol
	# 	error_old = error

	# 	while error > tol:
	# 		# Solve mechanical problem
	# 		salt.solve_mechanics()

	# 		# Compute error
	# 		error = salt.compute_error()
	# 		print(ite, error)

	# 		# Compute total strain
	# 		salt.compute_total_strain()

	# 		# Compute elastic strin
	# 		salt.compute_elastic_strain()

	# 		# Compute stress
	# 		salt.compute_stress()

	# 		# Compute creep
	# 		salt.compute_creep(time_handler)

	# 		# Increase iteration
	# 		ite += 1

	# 	# Update old strins
	# 	salt.update_old_strains()

	# 	# Save results
	# 	saver_eps_tot.record_average(salt.model_v.eps_tot, time_handler.time)

	# # Write results
	# saver_eps_tot.save(os.path.join(output_folder, "numeric"))







if __name__ == "__main__":
	main()