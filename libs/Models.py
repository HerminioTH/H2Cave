from fenics import *
import numpy as np
import sympy as sy
from Utils import *
from Elements import *
import abc


class BaseModel(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def __init__(self, fem_handler, bc_handler, settings):
		self.fem_handler = fem_handler
		self.bc_handler = bc_handler
		self.settings = settings

	@abc.abstractmethod
	def initialize(self, time_handler):
		pass

	@abc.abstractmethod
	def execute_model_pre(self, time_handler):
		pass

	@abc.abstractmethod
	def execute_iterative_procedure(self, time_handler):
		pass

	@abc.abstractmethod
	def execute_model_post(self, time_handler):
		pass

	# @abc.abstractmethod
	# def compute_stress(self):
	# 	pass

class ElasticModel(BaseModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
		self.__initialize_solution_vector()
		self.elastic_element = ElasticElement(self.fem_handler, self.settings, element_name="Elastic")

	def initialize(self, time_handler):
		pass

	def execute_model_pre(self, time_handler):
		# rhs vector
		self.bc_handler.update_BCs(time_handler)
		b = self.bc_handler.b

		# Stiffness matrix
		self.elastic_element.build_A()
		A = self.elastic_element.A

		# Solve linear system
		self.__solve_linear_system(A, b)

		# Compute strains
		self.elastic_element.compute_total_strain(self.u)
		self.elastic_element.compute_elastic_strain()

		# Compute stress
		self.elastic_element.compute_stress()


	def execute_iterative_procedure(self, time_handler):
		pass

	def execute_model_post(self, time_handler):
		pass

	def __solve_linear_system(self, A, b):
		[bc.apply(A, b) for bc in self.bc_handler.bcs]
		solve(A, self.u.vector(), b, "cg", "ilu")

	def __initialize_solution_vector(self):
		self.u = Function(self.fem_handler.V)
		self.u_k = Function(self.fem_handler.V)
		self.u.rename("Displacement", "m")
		self.u_k.rename("Displacement", "m")


class ViscoelasticModel(BaseModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
		self.__initialize_solution_vector()
		self.viscoelastic_element = ViscoelasticElement(self.fem_handler, self.settings)

	def initialize(self, time_handler):
		pass
		# self.__solve_elastic_model(time_handler)
		# self.viscoelastic_element.compute_total_strain(self.u)
		# self.viscoelastic_element.compute_elastic_strain()
		# self.viscoelastic_element.compute_stress()

	def execute_iterative_procedure(self, time_handler):
		pass

	def execute_model_pre(self, time_handler):
		# Compute constitutive matrices
		self.viscoelastic_element.compute_constitutive_matrices(time_handler.time_step)

		# rhs vector
		self.bc_handler.update_BCs(time_handler)
		self.viscoelastic_element.build_b()
		b = self.bc_handler.b + self.viscoelastic_element.b

		# Stiffness matrix
		self.viscoelastic_element.build_A()
		A = self.viscoelastic_element.A

		# Apply boundary conditions
		[bc.apply(A, b) for bc in self.bc_handler.bcs]

		# Solve instantaneous elastic problem
		solve(A, self.u.vector(), b, "cg", "ilu")

		# Compute strains
		self.viscoelastic_element.compute_total_strain(self.u)
		self.viscoelastic_element.compute_viscoelastic_strain()
		self.viscoelastic_element.compute_elastic_strain(eps_v=self.viscoelastic_element.eps_v)

		# Compute stress
		self.viscoelastic_element.compute_stress()

		# Update fields
		self.viscoelastic_element.update()

	def execute_model_post(self, time_handler):
		pass

	def __initialize_solution_vector(self):
		self.u = Function(self.fem_handler.V)
		self.u_k = Function(self.fem_handler.V)
		self.u.rename("Displacement", "m")
		self.u_k.rename("Displacement", "m")

	def __solve_elastic_model(self, time_handler):
		# rhs vector
		self.bc_handler.update_BCs(time_handler)
		b = self.bc_handler.b

		# Stiffness matrix
		self.viscoelastic_element.build_A_elastic()
		A = self.viscoelastic_element.A_elastic

		# Apply boundary conditions
		[bc.apply(A, b) for bc in self.bc_handler.bcs]

		# Solve instantaneous elastic problem
		solve(A, self.u.vector(), b, "cg", "ilu")



class BurgersModel(BaseModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
		self.__initialize_solution_vector()
		self.viscoelastic_element = ViscoelasticElement(self.fem_handler, self.settings)
		self.inelastic_elemens = []

	def add_inelastic_element(self, element_name):
		self.settings[self.element_name]["E"]

		self.inelastic_elements.append(inelastic_element)

	def initialize(self, time_handler):
		pass
		# self.__solve_elastic_model(time_handler)
		# self.viscoelastic_element.compute_total_strain(self.u)
		# self.viscoelastic_element.compute_elastic_strain()
		# self.viscoelastic_element.compute_stress()

	def execute_iterative_procedure(self, time_handler):
		pass

	def execute_model_pre(self, time_handler):
		# Compute constitutive matrices
		self.viscoelastic_element.compute_constitutive_matrices(time_handler.time_step)

		# rhs vector
		self.bc_handler.update_BCs(time_handler)
		self.viscoelastic_element.build_b()
		b = self.bc_handler.b + self.viscoelastic_element.b

		# Stiffness matrix
		self.viscoelastic_element.build_A()
		A = self.viscoelastic_element.A

		# Apply boundary conditions
		[bc.apply(A, b) for bc in self.bc_handler.bcs]

		# Solve instantaneous elastic problem
		solve(A, self.u.vector(), b, "cg", "ilu")

		# Compute strains
		self.viscoelastic_element.compute_total_strain(self.u)
		self.viscoelastic_element.compute_viscoelastic_strain()
		self.viscoelastic_element.compute_elastic_strain(eps_v=self.viscoelastic_element.eps_v)

		# Compute stress
		self.viscoelastic_element.compute_stress()

		# Update fields
		self.viscoelastic_element.update()

	def execute_model_post(self, time_handler):
		pass

	def __initialize_solution_vector(self):
		self.u = Function(self.fem_handler.V)
		self.u_k = Function(self.fem_handler.V)
		self.u.rename("Displacement", "m")
		self.u_k.rename("Displacement", "m")

	def __solve_elastic_model(self, time_handler):
		# rhs vector
		self.bc_handler.update_BCs(time_handler)
		b = self.bc_handler.b

		# Stiffness matrix
		self.viscoelastic_element.build_A_elastic()
		A = self.viscoelastic_element.A_elastic

		# Apply boundary conditions
		[bc.apply(A, b) for bc in self.bc_handler.bcs]

		# Solve instantaneous elastic problem
		solve(A, self.u.vector(), b, "cg", "ilu")