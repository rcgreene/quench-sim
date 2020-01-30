import numpy as np
import numpy.linalg as lin
import scipy as sp
import scipy.sparse as spar
from numba import jit
from scipy.spatial import KDTree
from scipy.optimize import root
import matplotlib.pyplot as plt

from node_finder import NodeFinder, NodeFinderSimple

@jit
def t_ind(a, b):
	#
	if a > b:
		return ((a + 1)*a)/2 + b
	else:
		return ((b + 1)*b)/2 + a

@jit
def phs(u, v):
	x_mag = lin.norm(u - v)
	return (x_mag**4)*np.log(x_mag) if x_mag != 0.0 else 0.

@jit
def phs_d(u, v):
	x_mag = lin.norm(u - v)
	return np.array([-4*v[i]*np.log(x_mag)*lin.norm(x_mag)**2 - v[i]*lin.norm(x_mag)**2 for i in range(2)])

@jit
def phs_2d(u, v):
	return np.array([[-5*(i == j)*lin.norm(u - v)**3 + 15*v[i]*v[j]*lin.norm(u - v) for i in range(2)] for j in range(2)])

@jit
def phs_3d(u, v):
	return np.array([
		[[ 15*(i == j)*lin.norm(u - v) + (v[i]*(j == k) + v[j]*(i == k))*15*lin.norm(u - v) - v[i]*v[j]*v[k]*15/lin.norm(u - v)
		 for i in range(2)]
		 for j in range(2)]
		 for k in range(2)])

class MonomialEvaluator:
	def __init__(self, dim, degree):
		# Class for enumerating polynomials up to a given degree and evaluating them
		self.dim = dim
		self.degree = degree
		self.set_up_list()

	def set_up_list(self):
		# make a list representing all monomials up to a given degree, represented by a list of the powers of the dim first degree monomials
		self.terms = [tuple([0,]*self.dim)]
		mono = np.array([0,]*(self.dim + 1))
		for i in range(self.degree):
			mono[:-1] = 0
			mono[-1] = i + 1
			while 1:
				self.terms.append(tuple(np.flip(np.array([mono[j + 1] - mono[j] for j in range(self.dim)]), axis=0)))
				if mono[1] == mono[-1]:
					break
				for j in range(self.dim - 1):
					if mono[j + 1] < mono[j + 2]:
						mono[j + 1] += 1
						break
					else:
						mono[j + 1] = 0
		self.term_num = len(self.terms)


	def evaluate_monomials(self, vec):
		#evaluate the monomial list and all of the first two derivatives
		mono = np.ones(self.term_num)
		#derivs = np.zeros((self.term_num, t_ind(self.dim, self.dim)))
		power_mat = np.array([vec**i for i in range(self.degree + 1)])
		for i in range(self.term_num):
			for j in range(self.dim):
				mono[i] *= power_mat[self.terms[i][j], j]

		return mono

	def evaluate_mono_derivs(self, vec, direct):
		# Evaluate the first derivatives in both directions, add based on weight provided by vector 'direct'
		mono = np.ones((2, self.term_num))
		power_mat = np.array([vec**i for i in range(self.degree + 1)])
		for k in range(2):
			for i in range(self.term_num):
				for j in range(self.dim):
					if j == k:
						mono[k, i] *= self.terms[i][j]
						if self.terms[i][j] == 0:
							break
						else:
							mono[k, i] *= power_mat[self.terms[i][j] - 1, j]
					else:
						mono[k, i] *= power_mat[self.terms[i][j], j]

		return mono[0]*direct[0] + mono[1]*direct[1]


class QuenchStencil:
	# Class used to set up the derivative stencils for the simulation
	def __init__(self, space, stencil_size, degree):
		node_out = space.get_node_info()
		self.nodes = node_out['nodes']
		self.pair_offset = node_out['pair_offset']
		self.dirichlet_num = node_out['dirichlet_num']
		self.boundary_num = node_out['boundary_num']
		self.outlet_num = node_out['outlet_num']
		self.lam = node_out['lambda']
		self.width = node_out['width']
		# Used to find nearest neighbors:
		self.nodes_mod = np.array([[np.cos(u[0]), np.sin(u[0]), u[1] + self.lam*u[0]] for u in self.nodes])

		self.init_boundary_op()
		self.init_stencil(stencil_size, degree)

	def init_boundary_op(self):

		@jit
		def boundary_op(r, storage):
			r_sq = r**2
			g = np.sqrt(r_sq + self.lam**2)/r
			storage[...] = 0
#			storage[1, 1] = 1/(g*r)
#			storage[0, 0] = -(self.lam/g)/(r_sq*g)
			storage[0, 1] = 1#g
			return storage

		del_s = self.lam*2*np.pi

		tmp = np.zeros((2,2))
		self.boundary_op = lambda r: boundary_op(r, tmp)

	def init_stencil(self, stencil_size, degree):
		NN_tree = KDTree(self.nodes_mod)
		NN_list = NN_tree.query(self.nodes_mod[self.dirichlet_num:], k=stencil_size)[1]
		self.evaluator = MonomialEvaluator(2, degree)

		row =  [[] for i in range(4)]
		col =  [[] for i in range(4)]
		data = [[] for i in range(4)]
		for i, NN in enumerate(NN_list):

			coords = np.array([self.nodes[j] - self.nodes[NN[0]] for j in NN])
			edge_list = {j: k < self.outlet_num for j, k in enumerate(NN) if k < self.dirichlet_num} # dictionary containing dirichlet nodes along with their type
			weights, wrap_weights = self.rbf_PHS(coords, edge_list, self.nodes[NN[0]][1], self.nodes[NN[0]][0])

			for j in range(4):
				row[j].extend([i]*stencil_size)
				col[j].extend(NN)
				data[j].extend(weights[j])
				if wrap_weights[j] != 0:
					row[j].append(i)
					col[j].append(self.nodes.shape[0])
					data[j].append(wrap_weights[j])

		self.deriv_stencils = [spar.csr_matrix(spar.coo_matrix( (data[j], (row[j], col[j])) )) for j in range(4)]
		self.deriv_adjoints = [u.transpose()[self.dirichlet_num:] for u in self.deriv_stencils]

	def get_direction(self, axis, s, alpha):
		if axis == 0:
			return np.array([1.0, 0.0])
		else:
			dir_vec = np.array([0.0, 1.0])#self.boundary_op(s + self.lam*alpha)[0]
			#dir_vec /= lin.norm(dir_vec)
			return dir_vec

	def rbf_PHS(self, GA_coords, edge_list, s, alpha):
		# Find stencil for first derivatives and homogeneous second derivatives using rbf phs
		# GA_coords are the coordinates relative to the central node. s is the s coordinate of the central node
		# edge_list is a _dictionary_ with keys corresponding to the nodes which are on an edge

		p_num = self.evaluator.term_num

		wrap_vec = np.zeros(GA_coords.shape[0])
		for i, x in enumerate(GA_coords):
			if x[0] < -np.pi:
				GA_coords[i, 0] += 2*np.pi
				GA_coords[i, 1] -= 2*np.pi*self.lam
				if i not in edge_list:
					wrap_vec[i] = -1.0
			if x[0] > np.pi:
				GA_coords[i, 0] -= 2*np.pi
				GA_coords[i, 1] += 2*np.pi*self.lam
				if i not in edge_list:
					wrap_vec[i] = 1.0

		eval_mat = np.zeros((GA_coords.shape[0] + p_num, GA_coords.shape[0] + p_num))

		deriv_mat = np.zeros((GA_coords.shape[0] + p_num, 4))

		eval_mat[:GA_coords.shape[0], :GA_coords.shape[0]] = np.array([[np.log(lin.norm(u - v))*lin.norm(u - v)**4 if lin.norm(u - v) != 0.0 else 0.0 for u in GA_coords] for v in GA_coords])
		
		if edge_list:
			# Correct eval_mat if there are boundary nodes present.
			for j in edge_list:
				direct = int(edge_list[j]) # cast bool as integer
				dir_vec = self.get_direction(direct, s + GA_coords[j, 1], alpha + GA_coords[j, 0])

				eval_mat[j, :GA_coords.shape[0]] = dir_vec[0]*np.array([4*(GA_coords[j, 0] - v[0])*np.log(lin.norm(GA_coords[j] - v))*lin.norm(GA_coords[j] - v)**2 
																		+ (GA_coords[j, 0] - v[0])*lin.norm(GA_coords[j] - v)**2 
																		if lin.norm(GA_coords[j] - v) != 0. else 0. for v in GA_coords])
				eval_mat[j, :GA_coords.shape[0]]+= dir_vec[1]*np.array([4*(GA_coords[j, 1] - v[1])*np.log(lin.norm(GA_coords[j] - v))*lin.norm(GA_coords[j] - v)**2
																		+ (GA_coords[j, 1] - v[1])*lin.norm(GA_coords[j] - v)**2 
																		if lin.norm(GA_coords[j] - v) != 0. else 0. for v in GA_coords])
				eval_mat[:GA_coords.shape[0], j] = eval_mat[j, :GA_coords.shape[0]]

			for i in edge_list:
				for j in edge_list:
					# Find direction vectors.
					direct_1 = int(edge_list[i])
					direct_2 = int(edge_list[j])
					dir_vecs = []
					for k, direct in [(i, direct_1), (j, direct_2)]:
						dir_vecs.append(self.get_direction(direct, s + GA_coords[k, 1], alpha + GA_coords[k, 0]))

					dir_matrix = np.outer(*dir_vecs)

					eval_mat[i, j] = 0
					for k in range(2):
						if i != j:
							eval_mat[i,j] -= 8*dir_matrix[k,k]*((GA_coords[i, k] - GA_coords[j, k])**2)*np.log(lin.norm(GA_coords[i] - GA_coords[j]))
							eval_mat[i,j] -= 6*dir_matrix[k,k]*((GA_coords[i, k] - GA_coords[j, k])**2)

							eval_mat[i,j] -= 4*dir_matrix[k,k]*np.log(lin.norm(GA_coords[j] - GA_coords[i]))*lin.norm(GA_coords[j] - GA_coords[i])**2
							eval_mat[i,j] -= dir_matrix[k,k]*lin.norm(GA_coords[j] - GA_coords[i])**2

						m = 1 - k
						if i != j:
							eval_mat[i,j] -= 8*dir_matrix[k,m]*((GA_coords[i, k] - GA_coords[j, k])*(GA_coords[i, m] - GA_coords[j, m]))*np.log(lin.norm(GA_coords[i] - GA_coords[j]))
							eval_mat[i,j] -= 6*dir_matrix[k,m]*((GA_coords[i, k] - GA_coords[j, k])*(GA_coords[i, m] - GA_coords[j, m]))
					eval_mat[j, i] = eval_mat[i, j]

		i_finder = np.zeros(2)

		for i in range(2):
			# First derivatives and homogeneous second derivatives
			i_finder[...] = 0
			i_finder[i] = 1
			index = self.evaluator.terms.index(tuple(i_finder))

			deriv_mat[GA_coords.shape[0] + index, i] = 1

			i_finder[...] = 0
			i_finder[i] = 2
			index = self.evaluator.terms.index(tuple(i_finder))
			deriv_mat[GA_coords.shape[0] + index, 2 + i] = 2

			deriv_mat[:GA_coords.shape[0], i] = np.array([ -4*v[i]*np.log(lin.norm(v))*lin.norm(v)**2 - v[i]*lin.norm(v)**2 if lin.norm(v) != 0. else 0. for v in GA_coords])
			deriv_mat[:GA_coords.shape[0], 2 + i] = np.array([ 4*np.log(lin.norm(v))*lin.norm(v)**2 + lin.norm(v)**2 +
														8*(v[i]**2)*np.log(lin.norm(v)) + 6*(v[i]**2) if lin.norm(v) != 0. else 0. for v in GA_coords])

			for j in edge_list:
				direct = int(edge_list[j])
				dir_vec = self.get_direction(direct, s + GA_coords[j, 1], alpha + GA_coords[j, 0])

				v = GA_coords[j]
				if lin.norm(v) != 0.0:
					deriv_mat[j, i] = -4*(lin.norm(v)**2)*np.log(lin.norm(v))*dir_vec[i] - dir_vec[i]*lin.norm(v)**2 - 8*np.dot(dir_vec, v)*v[i]*np.log(lin.norm(v)) - 6*np.dot(dir_vec, v)*v[i]
					deriv_mat[j, 2 + i] = 16*v[i]*dir_vec[i]*np.log(lin.norm(v)) + 8*np.dot(dir_vec, v)*np.log(lin.norm(v)) + 12*dir_vec[i]*v[i] 
					deriv_mat[j, 2 + i]+= 8*np.dot(dir_vec, v)*(v[i]**2)/(lin.norm(v)**2) + 6*np.dot(dir_vec, v)
		for i in range(GA_coords.shape[0]):
			if i in edge_list:
				direct = int(edge_list[i])
				dir_vec = self.get_direction(direct, s + GA_coords[i, 1], alpha + GA_coords[i, 0])
				poly_list = self.evaluator.evaluate_mono_derivs(GA_coords[i], dir_vec)
			else:
				poly_list = self.evaluator.evaluate_monomials(GA_coords[i])
			eval_mat[i, GA_coords.shape[0]:] = poly_list[...]
			eval_mat[GA_coords.shape[0]:, i] = poly_list[...]

		weights = lin.solve(eval_mat, deriv_mat)
		print 'max_error: ', np.max(np.abs(np.dot(eval_mat, weights) - deriv_mat))
		weights = weights[:GA_coords.shape[0]].T

		return weights, np.dot(weights, wrap_vec)

	def differentiate(self, i, v, target):
		target[...] = self.deriv_stencils[i].dot(v)

	def adjoint_diff(self, i, v, target):
		target[...] = self.deriv_stencils[i].dot(v)


class QuenchSim:
	# Class representing quenching on a coiled annulus superconductor
	def __init__(self, r_1, r_2, twist_num, h, stencil_size, degree, **kwargs):
		# initialize simulation
		if 'simple' in kwargs:
			if kwargs['simple']:
				self.space = NodeFinderSimple(r_1, r_2, h)
				self.SIMPLE = True
			else:
				self.space = NodeFinder(r_1, r_2, twist_num, h)
				self.SIMPLE = False
		else:
			self.space = NodeFinder(r_1, r_2, twist_num, h)
			self.SIMPLE = False
		self.stencil = QuenchStencil(self.space, stencil_size, degree)
		output = self.space.get_node_info()
		self.nodes = output['nodes']
		self.p_num = self.nodes.shape[0]
		self.lam = output['lambda']
		self.r = self.nodes[:, 1] + self.nodes[:, 0]*self.lam
		self.outlet_num = output['outlet_num']
		self.dirichlet_num = output['dirichlet_num']
		self.pair_num = self.dirichlet_num + output['pair_offset']
		self.pair_offset = output['pair_offset']

		self.r_1 = r_1

		self.eta = np.zeros((2, self.nodes.shape[0]))
		if not self.SIMPLE:
			self.U_val = np.zeros(self.nodes.shape[0] + 1)
		else:
			self.U_val = np.zeros(self.nodes.shape[0])
		self.U_deriv = np.zeros((4, self.nodes.shape[0]))
		self.eta_deriv = np.zeros((2, self.nodes.shape[0]))
		self.U_t = np.zeros(self.nodes.shape[0])
		self.U_grad = self.U_t.copy()

		self.process_keywords(kwargs)


	def process_keywords(self, kwargs):

		if 'outlet_slope' in kwargs:
			self.outlet_slope = kwargs['outlet_slope']
		else:
			self.outlet_slope = 0

		if 'eta_perp' in kwargs:
			self.eta[1] = kwargs['eta_perp']
		else:
			self.eta[1] = 0.0
		if 'eta_para' in kwargs:
			self.eta[0] = kwargs['eta_para']
		else:
			self.eta[0] = 1.0


	def get_derivs(self):

		for i in range(4):
			self.stencil.differentiate(i, self.U_val, self.U_deriv[i, self.dirichlet_num:])


	def get_U_t(self):
		self.get_derivs()
		self.U_t[...] = 0.0
		# Add up the individual terms. We use the product rule for derivatives.
		self.U_t += self.U_deriv[3]*self.eta[1]
		self.U_t += self.U_deriv[1]*self.eta[1]/self.r
#		self.U_t += self.eta_deriv[1]*self.U_deriv[1]
		self.U_t += self.U_deriv[2]*self.eta[0]/self.r**2
#		self.U_t += self.U_deriv[0]*self.eta_deriv[0]/self.r**2
		self.U_t -= 2*self.U_deriv[0]*self.eta[0]*self.lam/self.r**3


	def set_U(self, func):
		self.U_val[...] = 0
		self.U_val[self.dirichlet_num:self.p_num] = np.array([func(u) for u in self.nodes[self.dirichlet_num:]])
		self.U_val[:self.outlet_num] = self.outlet_slope
		if not self.SIMPLE:
			self.U_val[-1] = func(np.array([0.0, self.r_1 + 2*np.pi*self.lam])) - func([2*np.pi, self.r_1])


	def solve_U_t(self):

		def objective(U):
			self.U_val[(self.dirichlet_num + 1):self.p_num] = U[...]
			self.U_val[self.dirichlet_num] = 0.0
			self.get_U_t()
			return self.U_t[(self.dirichlet_num + 1):].copy()

		return root(objective, self.U_val[(self.dirichlet_num + 1):self.p_num].copy())


	def step_forward(self, t_step, step_num):
		rk_storage = np.zeros((4, self.U_t.shape[0]))

		for i in range(step_num):
			print i
			self.get_U_t()
			rk_storage[0] = self.U_t
			rk_storage[1] = self.U_val
			self.U_val[...] += self.U_t*t_step*.5
			self.get_U_t()
			rk_storage[2] = self.U_t
			self.U_val[...] = rk_storage[2]*t_step*.5 + rk_storage[1]
			self.get_U_t()
			rk_storage[3] = self.U_t
			self.U_val[...] = rk_storage[1] + t_step*self.U_t
			self.get_U_t()
			self.U_val = rk_storage[1] + t_step*(rk_storage[0]/6 + rk_storage[2]/3 + rk_storage[3]/3 + self.U_t/6)

	def plot_current(self):
		# plotting routine for the current, transformed according to the new coordinates.
		pass

