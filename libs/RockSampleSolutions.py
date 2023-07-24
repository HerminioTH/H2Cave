import numpy as np
import pandas as pd
import sympy as sy
import os

sec = 1.
minute = 60*sec
hour = 60*minute
day = 24*hour
kPa = 1e3
MPa = 1e6
GPa = 1e9

def double_dot(A, B):
	n, m = A.shape
	value = 0.0
	for i in range(n):
		for j in range(m):
			value += A[i,j]*B[j,i]
	return value

def trace(s):
	return s[0,0] + s[1,1] + s[2,2]

def build_stress(s_xx=0, s_yy=0, s_zz=0, s_xy=0, s_xz=0, s_yz=0):
	return np.array([s_xx, s_yy, s_zz, s_xy, s_xz, s_yz])

def voigt2tensor(s):
	return np.array([
			[s[0], s[3], s[4]],
			[s[3], s[1], s[5]],
			[s[4], s[5], s[2]],
		])

def tensor2voigt(s):
	return np.array([s[0,0], s[1,1], s[2,2], s[0,1], s[0,2], s[1,2]])

class TensorSaver():
	def __init__(self, output_folder, file_name):
		self.file_name = file_name
		self.output_folder = output_folder
		self.tensor_data = {
			"Time": [], "00": [], "01": [], "02": [], "10": [], "11": [], "12": [], "20": [], "21": [], "22": []
		}

	def save_results(self, time_list, eps_list):
		for time, eps in zip(time_list, eps_list):
			self.__record_values(eps, time)
		self.__save()

	def __record_values(self, tensor, t):
		self.tensor_data["Time"].append(t)
		for i in range(3):
			for j in range(3):
				self.tensor_data[f"{i}{j}"].append(tensor[i,j])

	def __save(self):
		if not os.path.exists(self.output_folder):
			os.makedirs(self.output_folder)
		self.df = pd.DataFrame(self.tensor_data)
		self.df.to_excel(os.path.join(self.output_folder, f"{self.file_name}.xlsx"))



class BaseSolution():
	def __init__(self, input_bc):
		self.__load_time_list(input_bc)
		self.__build_sigmas(input_bc)

	def __load_time_list(self, input_bc):
		self.time_list = np.array(input_bc["Time"]["timeList"])

	def __build_sigmas(self, input_bc):
		n = len(input_bc["sigma_xx"])
		sigma_xx = np.array(input_bc["sigma_xx"]).reshape((1, n))
		sigma_yy = np.array(input_bc["sigma_yy"]).reshape((1, n))
		sigma_zz = np.array(input_bc["sigma_zz"]).reshape((1, n))
		sigma_xy = np.array(input_bc["sigma_xy"]).reshape((1, n))
		sigma_yz = np.array(input_bc["sigma_yz"]).reshape((1, n))
		sigma_xz = np.array(input_bc["sigma_xz"]).reshape((1, n))
		self.sigmas = np.concatenate((sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz)).T
		self.sigma_0 = self.sigmas[0]

	def build_stress_increments(self):
		self.d_sigmas = []
		for i in range(1, len(self.sigmas)):
			self.d_sigmas.append(self.sigmas[i] - self.sigmas[i-1])
		self.d_sigmas = np.array(self.d_sigmas)

	def build_matrix(self, E, nu):
		lame = E*nu/((1+nu)*(1-2*nu))
		G = E/(2 +2*nu)
		x = 1 # Acho que aqui deveria ser dois
		self.C = np.array([
					[2*G + lame, 	lame, 		lame, 		0.,		0., 	0.],
					[lame, 			2*G + lame, lame, 		0.,		0., 	0.],
					[lame, 			lame, 		2*G + lame, 0., 	0., 	0.],
					[0., 			0., 		0., 		x*G,	0., 	0.],
					[0., 			0., 		0., 		0., 	x*G, 	0.],
					[0., 			0., 		0., 		0., 	0.,		x*G ],
				])
		self.D = np.linalg.inv(self.C)


class Elastic(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)
		self.build_stress_increments()
		self.build_matrix(self.E0, self.nu0)

	def __load_properties(self, input_model):
		self.E0 = input_model["Elements"]["Spring"]["E"]
		self.nu0 = input_model["Elements"]["Spring"]["nu"]

	def compute_strains(self):
		self.eps = []
		for i in range(len(self.time_list)):
			eps_value = voigt2tensor(np.dot(self.D, self.sigmas[i]))
			self.eps.append(eps_value)
		self.eps = np.array(self.eps)


class Viscoelastic(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)
		self.build_stress_increments()
		self.build_matrix(self.E0, self.nu0)

	def __load_properties(self, input_model):
		self.E0 = input_model["Elements"]["Spring"]["E"]
		self.nu0 = input_model["Elements"]["Spring"]["nu"]
		self.voigt_E = np.array([input_model["Elements"]["KelvinVoigt"]["E"]])
		self.voigt_eta = np.array([input_model["Elements"]["KelvinVoigt"]["eta"]])

	def A(self, t):
		try:
			value = np.zeros(len(t))
		except:
			value = 0
		for E, eta in zip(self.voigt_E, self.voigt_eta):
			value += (1 - np.exp(-E*t/eta))*self.E0/E
		return value

	def compute_strains(self):
		self.eps = []
		shape = self.d_sigmas[0].shape
		for i in range(0, len(self.time_list)):
			values = np.zeros(shape)
			A_list = self.A(self.time_list[i] - self.time_list[:i])
			A_list = A_list.reshape((len(A_list),1))
			values = A_list*self.d_sigmas[0:i]
			soma = np.sum(values, axis=0)
			eps_value = self.A(self.time_list[i])*voigt2tensor(np.dot(self.D, self.sigma_0))
			eps_value += voigt2tensor(np.dot(self.D, soma))
			self.eps.append(eps_value)
		self.eps = np.array(self.eps)

class DislocationCreep(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)

		self.eps_cr = np.zeros((3,3))
		self.eps_cr_old = np.zeros((3,3))
		self.eps_cr_rate = np.zeros((3,3))

	def __load_properties(self, input_model):
		self.A = input_model["Elements"]["DislocationCreep"]["A"]
		self.n = input_model["Elements"]["DislocationCreep"]["n"]
		self.T = input_model["Elements"]["DislocationCreep"]["T"]
		self.R = 8.32		# Universal gas constant
		self.Q = 51600  	# Creep activation energy, [J/mol]
		self.B = self.A*np.exp(-self.Q/(self.R*self.T))

	def update_internal_variables(self):
		self.eps_cr_old = self.eps_cr
		self.eps_cr_rate_old = self.eps_cr_rate

	def compute_eps_cr_rate(self, sigma):
		stress = voigt2tensor(sigma)
		s = stress - (1./3)*trace(stress)*np.eye(3)
		von_Mises = np.sqrt((3/2.)*double_dot(s, s))
		self.eps_cr_rate = self.B*(von_Mises**(self.n-1))*s

	def compute_eps_cr(self, i):
		t = self.time_list[i]
		t_old = self.time_list[i-1]
		dt = t - t_old
		self.compute_eps_cr_rate(self.sigmas[i])
		self.eps_cr = self.eps_cr_old + self.eps_cr_rate*dt
		self.eps_cr_old = self.eps_cr

	def compute_strains(self):
		self.eps = [self.eps_cr]
		for i in range(1, len(self.time_list)):
			self.compute_eps_cr(i)
			self.eps.append(self.eps_cr)
		self.eps = np.array(self.eps)


class PressureSolutionCreep(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)

		self.eps_cr = np.zeros((3,3))
		self.eps_cr_old = np.zeros((3,3))
		self.eps_cr_rate = np.zeros((3,3))

	def __load_properties(self, input_model):
		self.A = input_model["Elements"]["PressureSolutionCreep"]["A"]
		self.n = input_model["Elements"]["PressureSolutionCreep"]["n"]
		self.d = input_model["Elements"]["PressureSolutionCreep"]["d"]
		self.B = float(self.A)/(self.d**self.n)

	def update_internal_variables(self):
		self.eps_cr_old = self.eps_cr
		self.eps_cr_rate_old = self.eps_cr_rate

	def compute_eps_cr_rate(self, sigma):
		stress = voigt2tensor(sigma)
		s = stress - (1./3)*trace(stress)*np.eye(3)
		self.eps_cr_rate = self.B*s

	def compute_eps_cr(self, i):
		t = self.time_list[i]
		t_old = self.time_list[i-1]
		dt = t - t_old
		self.compute_eps_cr_rate(self.sigmas[i])
		self.eps_cr = self.eps_cr_old + self.eps_cr_rate*dt
		self.eps_cr_old = self.eps_cr

	def compute_strains(self):
		self.eps = [self.eps_cr]
		for i in range(1, len(self.time_list)):
			self.compute_eps_cr(i)
			self.eps.append(self.eps_cr)
		self.eps = np.array(self.eps)


class Damage(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)

		self.eps_d = np.zeros((3,3))
		self.eps_d_old = np.zeros((3,3))
		self.eps_d_rate = np.zeros((3,3))
		self.eps = [self.eps_d]
		self.D_list = [0.0]

	def __load_properties(self, input_model):
		self.A = input_model["Elements"]["Damage"]["A"]
		self.B = input_model["Elements"]["Damage"]["B"]
		self.n = input_model["Elements"]["Damage"]["n"]
		self.r = input_model["Elements"]["Damage"]["r"]
		self.nu = input_model["Elements"]["Damage"]["nu0"]
		# self.E = input_model["Elements"]["elasticity"]["E"]
		# self.G = self.E/(2*(1+self.nu))

	def update_internal_variables(self):
		self.eps_d_old = self.eps_d
		self.eps_d_rate_old = self.eps_d_rate

	def compute_eps_d(self, i):
		# Get time quantities
		t = self.time_list[i]/day
		t_old = self.time_list[i-1]/day
		dt = t - t_old

		# Compute deviatoric stress
		stress = voigt2tensor(self.sigmas[i])/MPa
		s = stress - (1./3)*trace(stress)*np.eye(3)

		# Compute von Mises stress
		von_Mises = np.sqrt((3/2.)*double_dot(s, s))

		# Compute mean stress
		sigma_m = (stress[0,0] + stress[1,1] + stress[2,2])/3

		# Compute damage equivalent stress
		sigma_star = von_Mises*np.sqrt((2/3)*(1 + self.nu + 3*(1 - 2*self.nu)*(sigma_m/von_Mises)**2))

		# Compute damage
		D_old = self.D_list[-1]
		D_k = D_old
		tol = 1e-7
		error = 2*tol
		ite = 0
		while error > tol:
			D_rate = (sigma_star/(self.B*(1 - D_k)))**self.r
			D = D_old + dt*D_rate
			error = abs(D_k - D)
			D_k = D
			ite += 1
		# print(ite, D, error)
		# D = 1 - (1 - t*(1+self.r)*(sigma_star/self.B)**self.r)**(1/(1+self.r))

		# Save damage
		self.D_list.append(D)

		# Compute damage strain rate
		self.eps_d_rate = (self.A*(von_Mises**(self.n-1))/((1 - D)**self.n))*s

		# Compute damage strain
		self.eps_d = self.eps_d_old + self.eps_d_rate*dt
		self.eps_d_old = self.eps_d.copy()

	def compute_strains(self):
		for i in range(1, len(self.time_list)):
			self.compute_eps_d(i)
			self.eps.append(self.eps_d)
		self.eps = np.array(self.eps)
		self.D_list = np.array(self.D_list)




class ViscoplasticDesai(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)
		self.__initialize_variables()
		self.__initialize_potential_function()
		self.qsi = 0.0

	def compute_strains(self):
		self.eps = [np.zeros((3,3))]
		for i in range(1, len(self.time_list)):
			dt = self.time_list[i] - self.time_list[i-1]
			# print(dt)
			# dt /= day
			stress_MPa = self.sigmas[i,:].copy()/MPa
			self.compute_yield_function(stress_MPa)
			# print("Fvp:", self.Fvp)
			
			ite = 1
			if self.Fvp <= 0:
				self.eps.append(self.eps[-1])
				self.alphas.append(self.alpha)
				self.alpha_qs.append(self.alpha_q)
				self.Fvp_list.append(self.Fvp)
			else:
				tol = 1e-6
				error = 2*tol
				maxiter = 20
				alpha_last = self.alpha
				while error > tol and ite < maxiter:
					strain_rate = self.__compute_strain_rate(stress_MPa)

					increment = double_dot(strain_rate, strain_rate)**0.5*dt
					self.qsi = self.qsi_old + increment


					increment_v = (strain_rate[0,0] + strain_rate[1,1] + strain_rate[2,2])*dt
					self.qsi_v = self.qsi_v_old + increment_v

					self.__update_kv(stress_MPa)
					self.__update_alpha()
					self.__update_alpha_q()

					error = abs(self.alpha - alpha_last)
					alpha_last = self.alpha
					self.compute_yield_function(stress_MPa)

					ite += 1
					if ite >= maxiter:
						print(f"Maximum number of iterations ({maxiter}) reached.")

				# string = (i, self.Fvp, strain_rate[0,0], strain_rate[1,1], strain_rate[2,2], (strain_rate[0,1]+strain_rate[0,2]+strain_rate[1,2]))
				# print("%i | %.4e | %.4e | %.4e | %.4e | %.4e"%string)
				# print(i, strain_rate[0,0], strain_rate[1,1], strain_rate[2,2], (strain_rate[0,1]+strain_rate[0,2]+strain_rate[1,2]))

				self.qsi_old = self.qsi
				self.qsi_v_old = self.qsi_v
				self.eps.append(self.eps[-1] + strain_rate*dt)
				self.alphas.append(self.alpha)
				self.alpha_qs.append(self.alpha_q)
				self.Fvp_list.append(self.Fvp)

			# print(self.alpha, self.Fvp, ite, strain_rate.flatten()[[0,4,8]])

		self.eps = np.array(self.eps)
		self.alphas = np.array(self.alphas)
		self.alpha_qs = np.array(self.alpha_qs)

	def __update_kv(self, stress_MPa):
		# sigma = stress_MPa[2]
		# self.k_v = -0.00085*sigma**2 + 0.015*sigma + 0.21
		# self.k_v = 0.18
		# coeffs = [2.39027657e-06, -4.54946293e-05, -6.57580943e-04,  4.99265504e-03,  1.81960713e-01, -6.45373053e-01]
		# coeffs = [-2.25330759e-07,  7.08080098e-06,  4.63967164e-05, -2.08478762e-03, -2.79699173e-02,  8.07033586e-01, -3.33527302e+00]
		# coeffs = [ 0.00859745, -0.34279313,  3.41342767]
		# coeffs = [ 0.00878372, -0.33816767,  3.28399277]
		# coeffs = [-7.66399407e-05,  3.77666296e-03, -6.25699148e-02,  4.23032481e-01, -8.83596888e-01]
		# func = np.poly1d(coeffs)
		# self.k_v = func(sigma)
		pass

	def __update_alpha(self):
		self.alpha = self.a_1 / (self.qsi**self.eta)
		# self.alpha = self.a_1 / (self.a_1/self.alpha_0 + self.qsi**self.eta)

	def __update_alpha_q(self):
		self.alpha_q = self.alpha + self.k_v*(self.alpha_0 - self.alpha)*(1 - self.qsi_v/self.qsi)

	def __compute_strain_rate(self, stress_MPa):
		n_flow = self.evaluate_flow_direction(stress_MPa, self.alpha_q)
		# print(n_flow)
		lmbda = self.mu_1*(self.Fvp/self.F0)**self.N_1
		strain_rate = lmbda*n_flow
		return strain_rate

	def __load_properties(self, input_model):
		self.mu_1 = input_model["Elements"]["ViscoplasticDesai"]["mu_1"]
		self.N_1 = input_model["Elements"]["ViscoplasticDesai"]["N_1"]
		self.n = input_model["Elements"]["ViscoplasticDesai"]["n"]
		self.a_1 = input_model["Elements"]["ViscoplasticDesai"]["a_1"]
		self.eta = input_model["Elements"]["ViscoplasticDesai"]["eta"]
		self.beta_1 = input_model["Elements"]["ViscoplasticDesai"]["beta_1"]
		self.beta = input_model["Elements"]["ViscoplasticDesai"]["beta"]
		self.m = input_model["Elements"]["ViscoplasticDesai"]["m"]
		self.gamma = input_model["Elements"]["ViscoplasticDesai"]["gamma"]
		self.k_v = input_model["Elements"]["ViscoplasticDesai"]["k_v"]
		self.sigma_t = input_model["Elements"]["ViscoplasticDesai"]["sigma_t"]
		self.alpha_0 = input_model["Elements"]["ViscoplasticDesai"]["alpha_0"]
		self.F0 = input_model["Elements"]["ViscoplasticDesai"]["F_0"]

	def __initialize_variables(self):
		self.alpha = self.alpha_0
		self.alpha_q = self.alpha_0
		self.alphas = [self.alpha]
		self.alpha_qs = [self.alpha_q]
		self.Fvp_list = [0]
		self.qsi_old = (self.a_1/self.alpha)**(1/self.eta)
		# self.qsi_old = 0.0
		self.qsi_v_old = self.qsi_old

	def __compute_stress_invariants(self, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz):
		I1 = s_xx + s_yy + s_zz# + self.sigma_t
		I2 = s_xx*s_yy + s_yy*s_zz + s_xx*s_zz - s_xy**2 - s_yz**2 - s_xz**2
		I3 = s_xx*s_yy*s_zz + 2*s_xy*s_yz*s_xz - s_zz*s_xy**2 - s_xx*s_yz**2 - s_yy*s_xz**2
		return I1, I2, I3

	def __compute_deviatoric_invariants(self, I1, I2, I3):
		J1 = np.zeros(I1.size) if type(I1) == np.ndarray else 0
		J2 = (1/3)*I1**2 - I2
		J3 = (2/27)*I1**3 - (1/3)*I1*I2 + I3
		return J1, J2, J3

	def __compute_Sr(self, J2, J3):
		return -(J3*np.sqrt(27))/(2*J2**1.5)

	def compute_yield_function(self, stress_MPa):
		I1, I2, I3 = self.__compute_stress_invariants(*stress_MPa)
		J1, J2, J3 = self.__compute_deviatoric_invariants(I1, I2, I3)
		if J2 == 0.0:
			self.Fvp = -100
		else:
			Sr = self.__compute_Sr(J2, J3)
			I1_star = I1 + self.sigma_t
			F1 = (-self.alpha*I1_star**self.n + self.gamma*I1_star**2)
			F2 = (np.exp(self.beta_1*I1_star) - self.beta*Sr)**self.m
			self.Fvp = J2 - F1*F2

	def __initialize_potential_function(self):
		# Stress components
		self.s_xx = sy.Symbol("s_xx")
		self.s_yy = sy.Symbol("s_yy")
		self.s_zz = sy.Symbol("s_zz")
		self.s_xy = sy.Symbol("s_xy")
		self.s_xz = sy.Symbol("s_xz")
		self.s_yz = sy.Symbol("s_yz")
		self.a_q = sy.Symbol("a_q")

		I1, I2, I3 = self.__compute_stress_invariants(self.s_xx, self.s_yy, self.s_zz, self.s_xy, self.s_xz, self.s_yz)
		J1, J2, J3 = self.__compute_deviatoric_invariants(I1, I2, I3)

		# Compute Lode's angle
		Sr = self.__compute_Sr(J2, J3)

		# Compute I_star
		I1_star = I1 #+ self.sigma_t

		# Potential function
		Q1 = (-self.a_q*I1_star**self.n + self.gamma*I1_star**2)
		Q2 = (sy.exp(self.beta_1*I1_star) - self.beta*Sr)**self.m
		Qvp = J2 - Q1*Q2

		variables = (self.s_xx, self.s_yy, self.s_zz, self.s_xy, self.s_xz, self.s_yz, self.a_q)
		self.dQdSxx = sy.lambdify(variables, sy.diff(Qvp, self.s_xx), "numpy")
		self.dQdSyy = sy.lambdify(variables, sy.diff(Qvp, self.s_yy), "numpy")
		self.dQdSzz = sy.lambdify(variables, sy.diff(Qvp, self.s_zz), "numpy")
		self.dQdSxy = sy.lambdify(variables, sy.diff(Qvp, self.s_xy), "numpy")
		self.dQdSxz = sy.lambdify(variables, sy.diff(Qvp, self.s_xz), "numpy")
		self.dQdSyz = sy.lambdify(variables, sy.diff(Qvp, self.s_yz), "numpy")

	def evaluate_flow_direction(self, stress, alpha_q):
		# Analytical derivatives
		dQdS = np.zeros((3,3))
		dQdS[0,0] = self.dQdSxx(*stress, alpha_q)
		dQdS[1,1] = self.dQdSyy(*stress, alpha_q)
		dQdS[2,2] = self.dQdSzz(*stress, alpha_q)
		dQdS[0,1] = dQdS[1,0] = self.dQdSxy(*stress, alpha_q)
		dQdS[0,2] = dQdS[2,0] = self.dQdSxz(*stress, alpha_q)
		dQdS[1,2] = dQdS[2,1] = self.dQdSyz(*stress, alpha_q)

		norm = np.linalg.norm(dQdS)
		dQdS = dQdS/norm
		return dQdS




class ViscoplasticDesai2(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)
		self.__initialize_variables()
		self.qsi = 0.0

	def compute_strains(self):
		self.eps = [np.zeros((3,3))]
		for i in range(1, len(self.time_list)):
			dt = self.time_list[i] - self.time_list[i-1]
			# print(dt)
			# dt /= day
			stress_MPa = self.sigmas[i,:].copy()/MPa
			self.compute_yield_function(stress_MPa)
			# print("Fvp:", self.Fvp)
			
			ite = 1
			if self.Fvp <= 0:
				self.eps.append(self.eps[-1])
				self.alphas.append(self.alpha)
				self.alpha_qs.append(self.alpha_q)
				self.Fvp_list.append(self.Fvp)
			else:
				tol = 1e-6
				error = 2*tol
				maxiter = 20
				alpha_last = self.alpha
				while error > tol and ite < maxiter:
					strain_rate = self.__compute_strain_rate(stress_MPa)

					increment = double_dot(strain_rate, strain_rate)**0.5*dt
					self.qsi = self.qsi_old + increment


					increment_v = (strain_rate[0,0] + strain_rate[1,1] + strain_rate[2,2])*dt
					self.qsi_v = self.qsi_v_old + increment_v

					self.__update_kv(stress_MPa)
					self.__update_alpha()
					self.__update_alpha_q()

					error = abs(self.alpha - alpha_last)
					alpha_last = self.alpha
					self.compute_yield_function(stress_MPa)

					ite += 1
					if ite >= maxiter:
						print(f"Maximum number of iterations ({maxiter}) reached.")

				# string = (i, self.Fvp, strain_rate[0,0], strain_rate[1,1], strain_rate[2,2], (strain_rate[0,1]+strain_rate[0,2]+strain_rate[1,2]))
				# print("%i | %.4e | %.4e | %.4e | %.4e | %.4e"%string)
				# print(i, strain_rate[0,0], strain_rate[1,1], strain_rate[2,2], (strain_rate[0,1]+strain_rate[0,2]+strain_rate[1,2]))

				self.qsi_old = self.qsi
				self.qsi_v_old = self.qsi_v
				self.eps.append(self.eps[-1] + strain_rate*dt)
				self.alphas.append(self.alpha)
				self.alpha_qs.append(self.alpha_q)
				self.Fvp_list.append(self.Fvp)

			# print(self.alpha, self.Fvp, ite, strain_rate.flatten()[[0,4,8]])

		self.eps = np.array(self.eps)
		self.alphas = np.array(self.alphas)
		self.alpha_qs = np.array(self.alpha_qs)

	def __update_kv(self, stress_MPa):
		# sigma = stress_MPa[2]
		# self.k_v = -0.00085*sigma**2 + 0.015*sigma + 0.21
		# self.k_v = 0.18
		# coeffs = [2.39027657e-06, -4.54946293e-05, -6.57580943e-04,  4.99265504e-03,  1.81960713e-01, -6.45373053e-01]
		# coeffs = [-2.25330759e-07,  7.08080098e-06,  4.63967164e-05, -2.08478762e-03, -2.79699173e-02,  8.07033586e-01, -3.33527302e+00]
		# coeffs = [ 0.00859745, -0.34279313,  3.41342767]
		# coeffs = [ 0.00878372, -0.33816767,  3.28399277]
		# coeffs = [-7.66399407e-05,  3.77666296e-03, -6.25699148e-02,  4.23032481e-01, -8.83596888e-01]
		# func = np.poly1d(coeffs)
		# self.k_v = func(sigma)
		pass

	def __update_alpha(self):
		self.alpha = self.a_1 / (self.qsi**self.eta)
		# self.alpha = self.a_1 / (self.a_1/self.alpha_0 + self.qsi**self.eta)

	def __update_alpha_q(self):
		self.alpha_q = self.alpha + self.k_v*(self.alpha_0 - self.alpha)*(1 - self.qsi_v/self.qsi)

	def __compute_strain_rate(self, stress_MPa):
		n_flow = self.evaluate_flow_direction(stress_MPa, self.alpha_q)
		# print(n_flow)
		lmbda = self.mu_1*(self.Fvp/self.F0)**self.N_1
		strain_rate = lmbda*n_flow
		return strain_rate

	def __load_properties(self, input_model):
		self.mu_1 = input_model["Elements"]["ViscoplasticDesai"]["mu_1"]
		self.N_1 = input_model["Elements"]["ViscoplasticDesai"]["N_1"]
		self.n = input_model["Elements"]["ViscoplasticDesai"]["n"]
		self.a_1 = input_model["Elements"]["ViscoplasticDesai"]["a_1"]
		self.eta = input_model["Elements"]["ViscoplasticDesai"]["eta"]
		self.beta_1 = input_model["Elements"]["ViscoplasticDesai"]["beta_1"]
		self.beta = input_model["Elements"]["ViscoplasticDesai"]["beta"]
		self.m = input_model["Elements"]["ViscoplasticDesai"]["m"]
		self.gamma = input_model["Elements"]["ViscoplasticDesai"]["gamma"]
		self.k_v = input_model["Elements"]["ViscoplasticDesai"]["k_v"]
		self.sigma_t = input_model["Elements"]["ViscoplasticDesai"]["sigma_t"]
		self.alpha_0 = input_model["Elements"]["ViscoplasticDesai"]["alpha_0"]
		self.F0 = input_model["Elements"]["ViscoplasticDesai"]["F_0"]

	def __initialize_variables(self):
		self.alpha = self.alpha_0
		self.alpha_q = self.alpha_0
		self.alphas = [self.alpha]
		self.alpha_qs = [self.alpha_q]
		self.Fvp_list = [0]
		self.qsi_old = (self.a_1/self.alpha)**(1/self.eta)
		# self.qsi_old = 0.0
		self.qsi_v_old = self.qsi_old

	def __compute_stress_invariants(self, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz):
		I1 = s_xx + s_yy + s_zz# + self.sigma_t
		I2 = s_xx*s_yy + s_yy*s_zz + s_xx*s_zz - s_xy**2 - s_yz**2 - s_xz**2
		I3 = s_xx*s_yy*s_zz + 2*s_xy*s_yz*s_xz - s_zz*s_xy**2 - s_xx*s_yz**2 - s_yy*s_xz**2
		return I1, I2, I3

	def __compute_deviatoric_invariants(self, I1, I2, I3):
		J1 = np.zeros(I1.size) if type(I1) == np.ndarray else 0
		J2 = (1/3)*I1**2 - I2
		J3 = (2/27)*I1**3 - (1/3)*I1*I2 + I3
		return J1, J2, J3

	def __compute_Sr(self, J2, J3):
		return -(J3*np.sqrt(27))/(2*J2**1.5)

	def compute_yield_function(self, stress_MPa):
		I1, I2, I3 = self.__compute_stress_invariants(*stress_MPa)
		J1, J2, J3 = self.__compute_deviatoric_invariants(I1, I2, I3)
		if J2 == 0.0:
			self.Fvp = -100
		else:
			Sr = self.__compute_Sr(J2, J3)
			I1_star = I1 + self.sigma_t
			F1 = (-self.alpha*I1_star**self.n + self.gamma*I1_star**2)
			F2 = (np.exp(self.beta_1*I1_star) - self.beta*Sr)**self.m
			self.Fvp = J2 - F1*F2
			print(I1_star, I1, J2, F2, F1)

	def evaluate_flow_direction(self, stress_MPa, alpha_q):
		s_xx = stress_MPa[0]
		s_yy = stress_MPa[1]
		s_zz = stress_MPa[2]
		s_xy = stress_MPa[3]
		s_xz = stress_MPa[4]
		s_yz = stress_MPa[5]

		I1, I2, I3 = self.__compute_stress_invariants(*stress_MPa)
		J1, J2, J3 = self.__compute_deviatoric_invariants(I1, I2, I3)

		I1_star = I1 + self.sigma_t
		Sr = self.__compute_Sr(J2, J3)

		F1 = (-self.alpha_q*I1**self.n + self.gamma*I1**2)
		F2 = (np.exp(self.beta_1*I1) - self.beta*Sr)
		F1_star = (-self.alpha_q*I1_star**self.n + self.gamma*I1_star**2)
		F2_star = (np.exp(self.beta_1*I1_star) - self.beta*Sr)
		F = J2 - F1_star*F2_star**self.m

		I1_aux = I1#_star

		dF1_dI1 = 2*self.gamma*I1_aux - self.alpha_q*self.n*I1_aux**(self.n-1)
		dF2m_dI1 = self.beta_1*self.m*np.exp(self.beta_1*I1_aux)*F2**(self.m-1)
		dF_dI1 = -(dF1_dI1*F2**self.m + F1*dF2m_dI1)
		dF_dJ2 = 1 - (3*np.sqrt(27)*self.beta*J3*self.m*F1*F2**(self.m-1))/(4*J2**(5/2))

		dF2_dJ2 = -(3*self.beta*J3*27**0.5)/(4*J2**(5/2))
		dF_dJ2 = 1 - F1*self.m*F2**(self.m-1)*dF2_dJ2
		dF_dJ3 = -self.m*F1*self.beta*np.sqrt(27)*F2**(self.m-1)/(2*J2**1.5)

		dI1_dSxx = 1.0
		dI1_dSyy = 1.0
		dI1_dSzz = 1.0
		dI1_dSxy = 0.0
		dI1_dSxz = 0.0
		dI1_dSyz = 0.0

		dI2_dSxx = s_yy + s_zz
		dI2_dSyy = s_xx + s_zz
		dI2_dSzz = s_xx + s_yy
		dI2_dSxy = -2*s_xy
		dI2_dSxz = -2*s_xz
		dI2_dSyz = -2*s_yz

		dI3_dSxx = s_yy*s_zz - s_yz**2
		dI3_dSyy = s_xx*s_zz - s_xz**2
		dI3_dSzz = s_xx*s_yy - s_xy**2
		dI3_dSxy = 2*(s_xz*s_yz - s_zz*s_xy)
		dI3_dSxz = 2*(s_xy*s_yz - s_yy*s_xz)
		dI3_dSyz = 2*(s_xz*s_xy - s_xx*s_yz)

		dJ2_dI1 = (2/3)*I1
		dJ2_dI2 = -1.0

		dJ2_dSxx = dJ2_dI1*dI1_dSxx + dJ2_dI2*dI2_dSxx
		dJ2_dSyy = dJ2_dI1*dI1_dSyy + dJ2_dI2*dI2_dSyy
		dJ2_dSzz = dJ2_dI1*dI1_dSzz + dJ2_dI2*dI2_dSzz
		dJ2_dSxy = dJ2_dI1*dI1_dSxy + dJ2_dI2*dI2_dSxy
		dJ2_dSxz = dJ2_dI1*dI1_dSxz + dJ2_dI2*dI2_dSxz
		dJ2_dSyz = dJ2_dI1*dI1_dSyz + dJ2_dI2*dI2_dSyz

		dJ3_dI1 = (2/9)*I1**2 - (1/3)*I2
		dJ3_dI2 = -(1/3)*I1
		dJ3_dI3 = 1.0

		dJ3_dSxx = dJ3_dI1*dI1_dSxx + dJ3_dI2*dI2_dSxx + dJ3_dI3*dI3_dSxx
		dJ3_dSyy = dJ3_dI1*dI1_dSyy + dJ3_dI2*dI2_dSyy + dJ3_dI3*dI3_dSyy
		dJ3_dSzz = dJ3_dI1*dI1_dSzz + dJ3_dI2*dI2_dSzz + dJ3_dI3*dI3_dSzz
		dJ3_dSxy = dJ3_dI1*dI1_dSxy + dJ3_dI2*dI2_dSxy + dJ3_dI3*dI3_dSxy
		dJ3_dSxz = dJ3_dI1*dI1_dSxz + dJ3_dI2*dI2_dSxz + dJ3_dI3*dI3_dSxz
		dJ3_dSyz = dJ3_dI1*dI1_dSyz + dJ3_dI2*dI2_dSyz + dJ3_dI3*dI3_dSyz

		dQdS = np.zeros((3,3))
		dQdS[0,0] = dF_dI1*dI1_dSxx + dF_dJ2*dJ2_dSxx + dF_dJ3*dJ3_dSxx
		dQdS[1,1] = dF_dI1*dI1_dSyy + dF_dJ2*dJ2_dSyy + dF_dJ3*dJ3_dSyy
		dQdS[2,2] = dF_dI1*dI1_dSzz + dF_dJ2*dJ2_dSzz + dF_dJ3*dJ3_dSzz
		dQdS[0,1] = dQdS[1,0] = dF_dI1*dI1_dSxy + dF_dJ2*dJ2_dSxy + dF_dJ3*dJ3_dSxy
		dQdS[0,2] = dQdS[2,0] = dF_dI1*dI1_dSxz + dF_dJ2*dJ2_dSxz + dF_dJ3*dJ3_dSxz
		dQdS[1,2] = dQdS[2,1] = dF_dI1*dI1_dSyz + dF_dJ2*dJ2_dSyz + dF_dJ3*dJ3_dSyz

		return dQdS




class ViscoPlastic_VonMises(BaseSolution):
	def __init__(self, settings):
		super().__init__(settings)
		self.__load_properties(settings)

	def __load_properties(self, settings):
		self.yield_stress = settings["vonmises"]["yield_stress"]
		self.eta = settings["vonmises"]["eta"]

	def compute_yield_function(self, sigma):
		stress = voigt2tensor(sigma)
		s = stress - (1./3)*trace(stress)*np.eye(3)
		von_Mises = np.sqrt((3/2.)*double_dot(s, s))
		f = von_Mises - self.yield_stress
		if f >= 0:
			self.yield_stress = von_Mises
		return f

	def compute_f_derivatives(self, sigma):
		dSigma = 0.001
		dFdS = np.zeros(6)
		for i in range(6):
			s = sigma.copy()
			s[i] += dSigma
			dFdS[i] = (self.compute_yield_function(sigma) - self.compute_yield_function(s))/dSigma
		return dFdS

	def ramp(self, f):
		return (f + abs(f))/2.

	def compute_strains(self):
		self.eps = [np.zeros((3,3))]
		for i in range(1, len(self.time_list)):
			f = self.compute_yield_function(self.sigmas[i])
			dFdS = self.compute_f_derivatives(self.sigmas[i])
			gamma = self.ramp(f)/self.eta
			self.eps.append(self.eps[-1] - gamma*voigt2tensor(dFdS))
			# if gamma > 0:
			# 	print(gamma*voigt2tensor(dFdS))
		self.eps = np.array(self.eps)




class ViscoPlasticVonMises(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)
		self.__initialize_variables()

	def __initialize_variables(self):
		self.alpha = self.alpha_0
		self.qsi = 0
		self.qsi_old = 0
		self.alphas = [self.alpha]
		self.Fvp = 0
		self.Fvp_list = [0]

	def __load_properties(self, input_model):
		self.yield_stress = input_model["Elements"]["ViscoplasticVonMises"]["yield_stress"]
		self.tau = input_model["Elements"]["ViscoplasticVonMises"]["tau"]
		self.eta = input_model["Elements"]["ViscoplasticVonMises"]["eta"]
		self.a = input_model["Elements"]["ViscoplasticVonMises"]["a"]
		self.alpha_0 = input_model["Elements"]["ViscoplasticVonMises"]["alpha_0"]

	def __compute_yield_function(self, sigma):
		stress = voigt2tensor(sigma)
		s = stress - (1./3)*trace(stress)*np.eye(3)
		q = np.sqrt((3/2.)*double_dot(s, s))
		self.Fvp = q - self.yield_stress*self.alpha_0/self.alpha

	def __evaluate_flow_direction_FD(self, sigma):
		dSigma = 0.00001
		dFdS = np.zeros(6)
		for i in range(6):
			s = sigma.copy()
			s[i] += dSigma
			self.__compute_yield_function(sigma)
			F_1 = self.Fvp
			self.__compute_yield_function(s)
			F_2 = self.Fvp
			dFdS[i] = -(F_1 - F_2)/dSigma
		return voigt2tensor(dFdS)

	def __evaluate_flow_direction_AN(self, sigma):
		stress = voigt2tensor(sigma)
		s = stress - (1./3)*trace(stress)*np.eye(3)
		q = np.sqrt((3/2.)*double_dot(s, s))
		dFdS = np.zeros(6)
		dFdS[0] = (2*stress[0,0] - stress[1,1] - stress[2,2])/q
		dFdS[1] = (2*stress[1,1] - stress[0,0] - stress[2,2])/q
		dFdS[2] = (2*stress[2,2] - stress[0,0] - stress[1,1])/q
		dFdS[3] = 3*stress[0,1]/(2*q)
		dFdS[4] = 3*stress[0,2]/(2*q)
		dFdS[5] = 3*stress[1,2]/(2*q)
		return voigt2tensor(dFdS)

	def __compute_strain_rate(self, sigma):
		n_flow = self.__evaluate_flow_direction_AN(sigma)
		# n_flow = self.__evaluate_flow_direction_FD(sigma)
		strain_rate = (self.Fvp/self.tau)*n_flow
		return strain_rate

	def __update_alpha(self):
		self.alpha = self.a / (self.a/self.alpha_0 + self.qsi**self.eta)

	def compute_strains(self):
		self.eps = [np.zeros((3,3))]
		for i in range(1, len(self.time_list)):
			dt = self.time_list[i] - self.time_list[i-1]
			self.__compute_yield_function(self.sigmas[i,:])
			
			if self.Fvp <= 0:
				self.eps.append(self.eps[-1])
				self.alphas.append(self.alpha)
				self.Fvp_list.append(self.Fvp)
			else:
				tol = 1e-6
				error = 2*tol
				maxiter = 20
				alpha_last = self.alpha
				ite = 1
				while error > tol and ite < maxiter:
					strain_rate = self.__compute_strain_rate(self.sigmas[i,:])

					increment = double_dot(strain_rate, strain_rate)**0.5*dt
					self.qsi = self.qsi_old + increment

					self.__update_alpha()

					error = abs(self.alpha - alpha_last)
					alpha_last = self.alpha
					self.__compute_yield_function(self.sigmas[i,:])

					ite += 1
					if ite >= maxiter:
						print(f"Maximum number of iterations ({maxiter}) reached.")

				self.qsi_old = self.qsi
				self.eps.append(self.eps[-1] + strain_rate*dt)
				self.alphas.append(self.alpha)
				self.Fvp_list.append(self.Fvp)

			# print(self.alpha, self.Fvp, ite, strain_rate.flatten()[[0,4,8]])
			# print(self.alpha, self.Fvp, ite, stress_MPa)

		self.eps = np.array(self.eps)
		self.alphas = np.array(self.alphas)



class ViscoPlasticDruckerPrager(BaseSolution):
	def __init__(self, input_model, input_bc):
		super().__init__(input_bc)
		self.__load_properties(input_model)
		self.__initialize_variables()

	def __initialize_variables(self):
		self.gamma = np.sin(self.theta_rad)*( 1/(3**0.5*(3 + np.sin(self.theta_rad))) + 1/(3**0.5*(3 - np.sin(self.theta_rad))) )
		self.alpha_0 = 3*self.c*np.cos(self.theta_rad)*( 1/(3**0.5*(3 + np.sin(self.theta_rad))) + 1/(3**0.5*(3 - np.sin(self.theta_rad))) )
		self.alpha = self.alpha_0
		self.alphas = [self.alpha]
		self.qsi = 0
		self.qsi_old = 0
		self.Fvp = 0
		self.Fvp_list = [0]

	def __load_properties(self, input_model):
		self.c = input_model["Elements"]["ViscoPlasticDruckerPrager"]["c"]
		self.theta_rad = np.radians(input_model["Elements"]["ViscoPlasticDruckerPrager"]["theta_deg"])
		self.alpha_1 = input_model["Elements"]["ViscoPlasticDruckerPrager"]["alpha_1"]
		self.alpha_2 = input_model["Elements"]["ViscoPlasticDruckerPrager"]["alpha_2"]
		self.k1 = input_model["Elements"]["ViscoPlasticDruckerPrager"]["k1"]
		self.k2 = input_model["Elements"]["ViscoPlasticDruckerPrager"]["k2"]
		self.beta = input_model["Elements"]["ViscoPlasticDruckerPrager"]["beta"]
		self.tau = input_model["Elements"]["ViscoPlasticDruckerPrager"]["tau"]

	def __compute_yield_function(self, sigma):
		stress = voigt2tensor(sigma)
		s = stress - (1./3)*trace(stress)*np.eye(3)
		I1 = trace(stress)
		J2 = (1/2)*double_dot(s, s)
		# print(-self.alpha*I1, np.sqrt(J2), self.k)
		self.Fvp = -self.gamma*I1 + np.sqrt(J2) - self.alpha

	def __evaluate_flow_direction(self, sigma):
		stress = voigt2tensor(sigma)
		s = stress - (1./3)*trace(stress)*np.eye(3)
		J2 = (1/2)*double_dot(s, s)
		I = np.eye(3)
		# dQdS = -self.gamma*I + (0.5/np.sqrt(J2))*s
		dQdS = -self.beta*I + (0.5/np.sqrt(J2))*s
		return dQdS

	def __compute_strain_rate(self, sigma):
		n_flow = self.__evaluate_flow_direction(sigma)
		strain_rate = (self.Fvp/self.tau)*n_flow
		return strain_rate

	def __update_hardening(self):
		# self.k = self.k0 + self.k1*(1 - np.exp(-self.k2*self.qsi))
		# value = 1 - (np.exp(-self.k2*self.qsi) + np.exp(-self.k3*self.qsi))/2
		# self.k = self.k0 + self.k1*value
		self.alpha = self.alpha_0
		self.alpha += self.alpha_1*(1 - np.exp(-self.k1*self.qsi))
		self.alpha += self.alpha_2*(1 - np.exp(-self.k2*self.qsi))

	def compute_strains(self):
		self.eps = [np.zeros((3,3))]
		self.qsi = 0
		for i in range(1, len(self.time_list)):
			dt = self.time_list[i] - self.time_list[i-1]
			stress_MPa = self.sigmas[i,:].copy()/MPa

			self.__compute_yield_function(stress_MPa)
			
			if self.Fvp <= 0:
				self.eps.append(self.eps[-1])
				self.alphas.append(self.alpha)
				self.Fvp_list.append(self.Fvp)
			else:
				tol = 1e-6
				error = 2*tol
				maxiter = 20
				alpha_last = self.alpha
				ite = 1
				while error > tol and ite < maxiter:
					strain_rate = self.__compute_strain_rate(stress_MPa)

					increment = double_dot(strain_rate, strain_rate)**0.5*dt
					self.qsi = self.qsi_old + increment

					self.__update_hardening()

					error = abs(self.alpha - alpha_last)
					alpha_last = self.alpha
					self.__compute_yield_function(stress_MPa)

					ite += 1
					if ite >= maxiter:
						print(f"Maximum number of iterations ({maxiter}) reached.")


				self.qsi_old = self.qsi
				self.eps.append(self.eps[-1] + strain_rate*dt)
				self.alphas.append(self.alpha)
				self.Fvp_list.append(self.Fvp)

			print(self.Fvp, self.qsi, self.alpha_0)
			# print(self.Fvp, self.alpha, self.alpha_0, self.alpha_1, self.qsi)
		self.eps = np.array(self.eps)
		self.alphas = np.array(self.alphas)


