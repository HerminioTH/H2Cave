from fenics import *
import os
import sys
import numpy as np
import shutil
import json

sys.path.append(os.path.join("..", "..", "libs"))
from GridHandler import GridHandler
from ResultsHandler import TensorSaver
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

def main():

	# Define settings
	settings = {
		"Paths" : {
			"Output": "output/case_0",
			"Grid": "../../grids/quarter_cylinder_0",
		},
		"Time" : {
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

	# Transient settings
	t = 0
	dt = settings["Time"]["timeStep"]
	t_final = settings["Time"]["finalTime"]
	theta = settings["Time"]["theta"]

	# Load grid
	grid_folder = os.path.join(*settings["Paths"]["Grid"].split("/"))
	geometry_name = "geom"
	grid = GridHandler(geometry_name, grid_folder)

	# Define output folder
	output_folder = os.path.join(*settings["Paths"]["Output"].split("/"))

	# Define function space
	V = VectorFunctionSpace(grid.mesh, "CG", 1)

	# Define Dirichlet boundary conditions
	bcs = []
	bcs.append(DirichletBC(V.sub(2), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["BOTTOM"]))
	bcs.append(DirichletBC(V.sub(1), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["SIDE_Y"]))
	bcs.append(DirichletBC(V.sub(0), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["SIDE_X"]))
	# bcs.append(DirichletBC(V.sub(0), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["OUTSIDE"]))
	# bcs.append(DirichletBC(V.sub(1), Constant(0.0), grid.boundaries, grid.dolfin_tags[grid.boundary_dim]["OUTSIDE"]))

	# Define measures
	dx = Measure("dx", domain=grid.mesh, subdomain_data=grid.subdomains)
	ds = Measure("ds", domain=grid.mesh, subdomain_data=grid.boundaries)

	# Define variational problem
	du = TrialFunction(V)
	d = du.geometric_dimension()  # space dimension
	v = TestFunction(V)

	# Define normal directions on mesh
	norm = FacetNormal(grid.mesh)
	v_n = dot(v, norm)

	# Define solution vector
	u = Function(V)
	u.rename("Displacement", "m")
	u_k = Function(V)
	u_k.rename("Displacement", "m")

	# # Define initial tensors
	TS = TensorFunctionSpace(grid.mesh, "DG", 0)

	# Create tensor savers
	saver_eps_e = TensorSaver("eps_e", dx)
	saver_eps_tot = TensorSaver("eps_tot", dx)
	saver_eps_v = TensorSaver("eps_v", dx)
	saver_eps_cr = TensorSaver("eps_cr", dx)
	saver_eps_cr_rate = TensorSaver("eps_cr_rate", dx)
	saver_stress = TensorSaver("stress", dx)

	# Define viscoelastic model
	model_v = ViscoElasticModel(settings["Viscoelastic"], theta, du, v, dx, TS)
	model_v.compute_matrices(dt)

	# Define creep model (Power-Law)
	model_c = CreepDislocation(settings["DislocationCreep"], theta, model_v.C0, du, v, dx, TS)

	# Initialize matrix
	model_v.build_A_elastic()
	A = model_v.A_elastic

	# Apply Neumann boundary conditions
	L_bc = -axial_stress(t)*v_n*ds(grid.dolfin_tags[grid.boundary_dim]["TOP"])
	b_bc = assemble(L_bc)
	b = b_bc

	# Solve instantaneous elastic problem
	[bc.apply(A, b) for bc in bcs]
	solve(A, u.vector(), b, "cg", "ilu")


	# Compute total strain
	model_v.compute_total_strain(u)

	# Compute elastic strain
	eps_ie = model_c.eps_cr
	model_v.compute_elastic_strain(eps_ie)

	# Compute stress
	model_v.compute_stress()

	# Record tensors
	saver_eps_e.record_average(model_v.eps_e, t)
	saver_eps_v.record_average(model_v.eps_v, t)
	saver_eps_cr.record_average(model_c.eps_cr, t)
	saver_eps_cr_rate.record_average(model_c.eps_cr_rate, t)
	saver_eps_tot.record_average(model_v.eps_tot, t)
	saver_stress.record_average(model_v.stress, t)

	# Compute viscous strain
	eps_ie_old = model_c.eps_cr_old
	model_v.compute_viscous_strain(eps_ie, eps_ie_old)

	# Update total strain
	model_v.update()

	# Include viscous terms for transient simulation
	model_v.build_A()
	# A += model_v.A

	while t < t_final:
		# Update time
		t += dt
		print(t)

		# Apply Neumann boundary conditions
		L_bc = -axial_stress(t)*v_n*ds(grid.dolfin_tags[grid.boundary_dim]["TOP"])
		b_bc = assemble(L_bc)

		# Iteration settings
		ite = 0
		tol = 1e-9
		error = 2*tol
		error_old = error

		# Compute creep strain
		model_c.compute_creep_strain(model_v.stress, dt)

		while error > tol:
			b = 0

			# # Build creep rhs
			model_c.build_b()
			b += model_c.b

			# Build viscoelastic rhs
			model_v.build_b(model_c.eps_cr, model_c.eps_cr_old)
			b += model_v.b

			# Boundary condition rhs
			b += b_bc

			# Apply Dirichlet boundary conditions
			[bc.apply(A, b) for bc in bcs]

			# Solve linear system
			# solve(A, u.vector(), b)
			solve(A, u.vector(), b, "cg", "ilu")

			# Compute error
			error = compute_error(u, u_k)
			print(ite, error)

			# Update error
			error_old = error

			# Update solution
			u_k.assign(u)

			# Compute total strain
			model_v.compute_total_strain(u)

			# Compute elastic strain
			model_v.compute_elastic_strain(model_c.eps_cr)

			# Compute stress
			model_v.compute_stress()

			# Compute creep strain
			model_c.compute_creep_strain(model_v.stress, dt)

			# Increase iteration
			ite += 1

		# Compute viscous strain
		model_v.compute_viscous_strain(model_c.eps_cr, model_c.eps_cr_old)

		# Update old variables
		model_v.update()
		model_c.update()

		# Record tensors
		saver_eps_e.record_average(model_v.eps_e, t)
		saver_eps_v.record_average(model_v.eps_v, t)
		saver_eps_cr.record_average(model_c.eps_cr, t)
		saver_eps_cr_rate.record_average(model_c.eps_cr_rate, t)
		saver_eps_tot.record_average(model_v.eps_tot, t)
		saver_stress.record_average(model_v.stress, t)

	# 	# Save results
	# 	# vtk_displacement << (u, t)
	# 	# vtk_stress << (stress, t)
	# 	# vtk_strain_e << (eps_e, t)
	# 	# vtk_strain_v << (eps_v, t)
	# 	# vtk_strain_tot << (eps_tot, t)
	# 	# vtk_strain_cr << (eps_cr, t)
	# 	# vtk_strain_cr_rate << (eps_cr_rate, t)

	# Save tensor results
	saver_eps_e.save(os.path.join(output_folder, "numeric"))
	saver_eps_v.save(os.path.join(output_folder, "numeric"))
	saver_eps_cr.save(os.path.join(output_folder, "numeric"))
	saver_eps_cr_rate.save(os.path.join(output_folder, "numeric"))
	saver_eps_tot.save(os.path.join(output_folder, "numeric"))
	saver_stress.save(os.path.join(output_folder, "numeric"))

	# Copy .msh mesh to output_folder
	source = os.path.join(grid_folder, geometry_name+".msh")
	destination = output_folder
	shutil.copy(source, destination)
	shutil.copy(__file__, os.path.join(destination, "copy.py"))

if __name__ == "__main__":
	main()