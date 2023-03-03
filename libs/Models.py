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
	# 	pass", "m")

class MechanicsModel(BaseModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
		self.__initialize_solution_vector()

	def __initialize_solution_vector(self):
		self.u = Function(self.fem_handler.V)
		self.u_k = Function(self.fem_handler.V)
		# self.u_k = interpolate(Constant((0.0, 0.0, 0.0)), self.fem_handler.V)
		self.u.rename("Displacement", "m")
		# self.u_k.rename("Displacement", "m")

	def update_solution_vector(self):
		self.u_k.assign(self.u)
		# self.u_k.vector().set_local(self.u.vector())
		# self.u_k = self.u.copy()
		# self.u_k.assign(self.u)
		# self.u_k = self.u.vector().array()

class ElasticModel(MechanicsModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
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


class ViscoelasticModel(MechanicsModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
		self.viscoelastic_element = ViscoelasticElement(self.fem_handler, self.settings)

	def initialize(self, time_handler):
		pass

	def execute_model_pre(self, time_handler):
		# Compute constitutive matrices
		self.viscoelastic_element.compute_constitutive_matrices(time_handler.time_step)

		# Assemble stiffness matrix
		self.viscoelastic_element.build_A()

		# rhs vector
		self.bc_handler.update_BCs(time_handler)
		# rhs vector
		self.viscoelastic_element.build_b()
		b = self.bc_handler.b + self.viscoelastic_element.b

		# Apply boundary conditions
		[bc.apply(self.viscoelastic_element.A, b) for bc in self.bc_handler.bcs]

		# Solve instantaneous elastic problem
		solve(self.viscoelastic_element.A, self.u.vector(), b, "cg", "ilu")

		# Compute strains
		self.viscoelastic_element.compute_total_strain(self.u)
		self.viscoelastic_element.compute_viscoelastic_strain()
		self.viscoelastic_element.compute_elastic_strain()

		# Compute stress
		self.viscoelastic_element.compute_stress()

		# Update fields
		self.viscoelastic_element.update()

	def execute_iterative_procedure(self, time_handler):
		pass

	def execute_model_post(self, time_handler):
		pass

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


class MaxwellModel(MechanicsModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
		self.elastic_element = ElasticElement(self.fem_handler, self.settings, element_name="Elastic")
		self.dashpot_element = DashpotElement(self.fem_handler, self.settings, element_name="Dashpot")

	def initialize(self, time_handler):
		# Stiffness matrix
		self.elastic_element.build_A()

	def execute_model_pre(self, time_handler):
		# Update boundary condition
		self.bc_handler.update_BCs(time_handler)

	def execute_iterative_procedure(self, time_handler):
		# rhs vector
		self.dashpot_element.build_b(self.elastic_element.C0)
		b = self.bc_handler.b + self.dashpot_element.b

		# Solve linear system
		self.__solve_linear_system(self.elastic_element.A, b)
		# print(np.linalg.norm(self.u.vector()), np.linalg.norm(self.u_k.vector()))

		# Compute total strain
		self.elastic_element.compute_total_strain(self.u)

		# Compute elastic strain
		self.elastic_element.compute_elastic_strain(eps_ie=self.dashpot_element.eps_ie)

		# Compute stress
		self.elastic_element.compute_stress()

		# Compute inelastic strain
		self.dashpot_element.compute_viscous_strain(self.elastic_element.stress, time_handler.time_step)

	def execute_model_post(self, time_handler):
		self.dashpot_element.update()

	def __solve_linear_system(self, A, b):
		[bc.apply(A, b) for bc in self.bc_handler.bcs]
		solve(A, self.u.vector(), b, "cg", "ilu")



class BurgersModel(MechanicsModel):
	def __init__(self, fem_handler, bc_handler, settings):
		super().__init__(fem_handler, bc_handler, settings)
		self.viscoelastic_element = ViscoelasticElement(self.fem_handler, self.settings, element_name="Viscoelastic")
		self.inelastic_elements = []

	def add_inelastic_element(self, inelastic_element):
		self.inelastic_elements.append(inelastic_element)

	def initialize(self, time_handler):
		pass

	def execute_model_pre(self, time_handler):
		# Compute constitutive matrices
		self.viscoelastic_element.compute_constitutive_matrices(time_handler.time_step)

		# Stiffness matrix
		self.viscoelastic_element.build_A()
		A = self.viscoelastic_element.A

		# rhs vector due to boundary conditions
		self.bc_handler.update_BCs(time_handler)

	def execute_iterative_procedure(self, time_handler):
		# Get inelastic strains from dashpots
		eps_ie = self.__get_eps_ie()
		eps_ie_old = self.__get_eps_ie_old()

		# rhs vector
		self.viscoelastic_element.build_b(eps_ie=eps_ie, eps_ie_old=eps_ie_old)
		b = self.bc_handler.b + self.viscoelastic_element.b

		for ie_element in self.inelastic_elements:
			ie_element.build_b(self.viscoelastic_element.C0)
			b += ie_element.b

		# Apply boundary conditions
		[bc.apply(self.viscoelastic_element.A, b) for bc in self.bc_handler.bcs]

		# Solve instantaneous elastic problem
		solve(self.viscoelastic_element.A, self.u.vector(), b, "cg", "ilu")

		# Compute strains
		self.viscoelastic_element.compute_total_strain(self.u)
		self.viscoelastic_element.compute_viscoelastic_strain(eps_ie=eps_ie, eps_ie_old=eps_ie_old)
		self.viscoelastic_element.compute_elastic_strain(eps_ie=eps_ie)

		# Compute stress
		self.viscoelastic_element.compute_stress()

		# Update inelastic strains
		self.__compute_eps_ie(self.viscoelastic_element.stress, time_handler)


	def execute_model_post(self, time_handler):
		# Update fields
		self.viscoelastic_element.update()
		self.__update_eps_ie()

	def __get_eps_ie_old(self):
		eps_ie_old = 0
		for ie_element in self.inelastic_elements:
			eps_ie_old += ie_element.eps_ie_old
		if eps_ie_old == 0:
			return None
		else:
			return eps_ie_old

	def __get_eps_ie(self):
		eps_ie = 0
		for ie_element in self.inelastic_elements:
			eps_ie += ie_element.eps_ie
		if eps_ie == 0:
			return None
		else:
			return eps_ie

	def __compute_eps_ie(self, stress, time_handler):
		for ie_element in self.inelastic_elements:
			ie_element.compute_viscous_strain(stress, time_handler.time_step)

	def __update_eps_ie(self):
		for ie_element in self.inelastic_elements:
			ie_element.update()
