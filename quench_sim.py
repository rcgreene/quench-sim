import numpy as np
import numpy.linalg as lin
import scipy as sp
import scipy.sparse as spar
from scipy.sparse.linalg import spsolve
from numba import jit
from scipy.spatial import KDTree
from scipy.special import gammainc
from scipy.optimize import root
import matplotlib.pyplot as plt
from rbf_ga import MonomialEvaluator, stencilMaker, get_max_degree
from hermite_rbf import hermite_laplace, hermite_deriv

from node_finder import NodeFinder, NodeFinderSimple, NodeStepper

INTERIOR    = 0
LOWER_BOUND = 1
UPPER_BOUND = 2
LOWER_CUSP  = 3
UPPER_CUSP  = 4
INLET_CUSP  = 5
OUTLET_CUSP = 6


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

class GeometryHelper:
	def __init__(self, lam_val = .0, theta_factor = 1.0):
		self.lam_val = lam_val
		self.cc_mat = np.array([[1., 0.],[self.lam_val, 1.]])
		self.theta_factor = theta_factor # correction factor to make distances in charts comparable in all directions

	def get_local_tangents(self, r, theta):
		# Get the principal directions for the local chart. Tangents always have the same lengths.
		tans = np.array([
						[self.lam_val, r],
						[1., -self.lam_val/r]])
		tans /= lin.norm(tans, axis=1, keepdims=True)
		tans = tans @ self.get_rotation(theta)
		tans[0] /= self.theta_factor
		return tans

	def get_rotation(self, theta):
		return np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]]).T

	def get_local_chart(self, center, p_list):
		# Convert points to a local euclidean map in which x = x_|| and y = x_+. Distances along the parallel axis
		# Are reduced by a correction factor
		theta, r = self.cc_mat @ center
		new_p_list = p_list @ self.cc_mat.T
		tans = self.get_local_tangents(r, theta)
		euc_center = np.array([r*np.cos(theta), r*np.sin(theta)])
		new_p_list = np.array([[u[1]*np.cos(u[0]), u[1]*np.sin(u[0])] - euc_center for u in new_p_list])
		new_p_list = new_p_list @ tans.T
		wrap_list = np.array([np.sign(p_list[i][0] - theta)*(np.abs(p_list[i][0] - theta) > np.pi) for i in range(len(new_p_list))])
		print(np.sum(wrap_list))
		#if np.sum(wrap_list) > 10:
		#	x = new_p_list[:,0]
		#	y = new_p_list[:,1]
		#	fig = plt.figure()
		#	ax = fig.add_subplot(111)
		#	ax.scatter(x, y)
		#	plt.show()
		#	x = (p_list @ self.cc_mat.T)[:,0]
		#	y = (p_list @ self.cc_mat.T)[:,1]
		#	fig = plt.figure()
		#	ax = fig.add_subplot(111)
		#	ax.scatter(np.cos(x)*y, np.sin(x)*y)
		#	plt.show()
		return new_p_list, wrap_list

	def get_tangent_derivative(self, center):
		# Get the coefficients to take into account the rotation of the x_|| and x_+ directions.
		theta, r = self.cc_mat @ center
		lam_factor = 1/(self.lam_val**2 + r**2)**(3./2)
		d_para = (-r**2 - 2*self.lam_val**2)*lam_factor
		d_perp = self.lam_val*r*lam_factor
		d_perp += -self.lam_val/(r*np.sqrt(self.lam_val**2 + r**2))
		return d_para, d_perp

	def get_edge_vec(self, edge_point, center, TYPE=0):
		theta,   r   = self.cc_mat @ center
		theta_2, r_2 = self.cc_mat @ edge_point
		tans = self.get_local_tangents(r, theta)
		tans[0] *= self.theta_factor
		edge_vec = self.get_local_tangents(r_2, theta_2)[0]*self.theta_factor
		return edge_vec @ tans.T




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
		self.rbf_maker = stencilMaker(stencil_size)
		# Used to find nearest neighbors:
		self.nodes_mod = np.array([[np.cos(u[0])*2*np.pi*(u[1] + self.lam*u[0]), np.sin(u[0])*2*np.pi*(u[1] + self.lam*u[0]), np.sqrt(np.pi**2 - 1)*(u[1] + self.lam*u[0])] for u in self.nodes])

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
		chart_finder = GeometryHelper(self.lam, 1.0)

		row =  [[] for i in range(4)]
		col =  [[] for i in range(4)]
		data = [[] for i in range(4)]
		for i, NN in enumerate(NN_list):

			p_list = np.array([self.nodes[j] for j in NN])
			coords, wrap_list = chart_finder.get_local_chart(p_list[0], p_list)
			edge_list = {j: chart_finder.get_edge_vec(self.nodes[k], p_list[0]) for j, k in enumerate(NN) if k < self.dirichlet_num} # dictionary containing dirichlet nodes along with their type
			for j in edge_list:
				wrap_list[j] = 0.0
			weights = self.rbf_maker.rbf_ga(coords, edge_list)
			wrap_weights = weights @ wrap_list
			d_para, d_perp = chart_finder.get_tangent_derivative(p_list[0])
			weights[2] /= chart_finder.theta_factor
			weights[0] /= chart_finder.theta_factor
			weights[2] += weights[1]*d_para
			weights[3] += weights[0]*d_perp

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

	def rbf_PHS(self, GA_coords, edge_list, s, alpha, wrap_vec):
		# Find stencil for first derivatives and homogeneous second derivatives using rbf phs
		# GA_coords are the coordinates relative to the central node. s is the s coordinate of the central node
		# edge_list is a _dictionary_ with keys corresponding to the nodes which are on an edge

		p_num = self.evaluator.term_num

		#wrap_vec = np.zeros(GA_coords.shape[0])
		#for i, x in enumerate(GA_coords):
		#	if x[0] < -np.pi:
		#		GA_coords[i, 0] += 2*np.pi
		#		GA_coords[i, 1] -= 2*np.pi*self.lam
		#		if i not in edge_list:
		#			wrap_vec[i] = -1.0
		#	if x[0] > np.pi:
		#		GA_coords[i, 0] -= 2*np.pi
		#		GA_coords[i, 1] += 2*np.pi*self.lam
		#		if i not in edge_list:
		#			wrap_vec[i] = 1.0

		eval_mat = np.zeros((GA_coords.shape[0] + p_num, GA_coords.shape[0] + p_num))

		deriv_mat = np.zeros((GA_coords.shape[0] + p_num, 4))

		eval_mat[:GA_coords.shape[0], :GA_coords.shape[0]] = np.array([[np.log(lin.norm(u - v))*lin.norm(u - v)**4 if lin.norm(u - v) != 0.0 else 0.0 for u in GA_coords] for v in GA_coords])
		
		if edge_list:
			# Correct eval_mat if there are boundary nodes present.
			for j in edge_list:
#				direct = int(edge_list[j]) # cast bool as integer
				dir_vec = edge_list[j]

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
#					direct_1 = int(edge_list[i])
#					direct_2 = int(edge_list[j])
					dir_vecs = [edge_list[i], edge_list[j]]
#					for k, direct in [(i, direct_1), (j, direct_2)]:
#						dir_vecs.append(self.get_direction(direct, s + GA_coords[k, 1], alpha + GA_coords[k, 0]))

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
#				direct = int(edge_list[j])
				dir_vec = edge_list[j]#self.get_direction(direct, s + GA_coords[j, 1], alpha + GA_coords[j, 0])

				v = GA_coords[j]
				if lin.norm(v) != 0.0:
					deriv_mat[j, i] = -4*(lin.norm(v)**2)*np.log(lin.norm(v))*dir_vec[i] - dir_vec[i]*lin.norm(v)**2 - 8*np.dot(dir_vec, v)*v[i]*np.log(lin.norm(v)) - 6*np.dot(dir_vec, v)*v[i]
					deriv_mat[j, 2 + i] = 16*v[i]*dir_vec[i]*np.log(lin.norm(v)) + 8*np.dot(dir_vec, v)*np.log(lin.norm(v)) + 12*dir_vec[i]*v[i] 
					deriv_mat[j, 2 + i]+= 8*np.dot(dir_vec, v)*(v[i]**2)/(lin.norm(v)**2) + 6*np.dot(dir_vec, v)

		for i in range(GA_coords.shape[0]):
			if i in edge_list:
#				direct = int(edge_list[i])
				dir_vec = edge_list[j]#self.get_direction(direct, s + GA_coords[i, 1], alpha + GA_coords[i, 0])
				poly_list = self.evaluator.evaluate_mono_derivs(GA_coords[i], dir_vec)
			else:
				poly_list = self.evaluator.evaluate_monomials(GA_coords[i])
			eval_mat[i, GA_coords.shape[0]:] = poly_list[...]
			eval_mat[GA_coords.shape[0]:, i] = poly_list[...]

		weights = lin.solve(eval_mat, deriv_mat)
		print('max_error: ', np.max(np.abs(np.dot(eval_mat, weights) - deriv_mat)))
		weights = weights[:GA_coords.shape[0]].T

		return weights, np.dot(weights, wrap_vec)

	def differentiate(self, i, v, target):
		target[...] = self.deriv_stencils[i].dot(v)

	def adjoint_diff(self, i, v, target):
		target[...] = self.deriv_stencils[i].dot(v)

class QuenchHermite:
	def __init__(self, space, stencil_size):
		data = space.get_spatial_data()
		self.evaluator = MonomialEvaluator(2, get_max_degree(stencil_size//2) - 1)
		print(get_max_degree(stencil_size//2))
		self.nodes = np.array(data['nodes'])
		self.lam = data['lambda']
		self.nodes_mod = np.array([[np.cos(u[0])*2*np.pi*(u[1] + self.lam*u[0]), np.sin(u[0])*2*np.pi*(u[1] + self.lam*u[0]), np.sqrt(np.pi**2 - 1)*(u[1] + self.lam*u[0])] for u in self.nodes])
		self.twist_num = data['twist_num']
		self.r_1 = data['r_1']
		self.r_2 = data['r_2']
		self.width = self.r_2 - self.r_1
		self.nodes[:, 0] /= 2*np.pi
		self.nodes[:, 1] -= self.r_1
		self.nodes[:, 1] /= self.width
		self.cusp_points = {}
		self.cusp_points[INLET_CUSP] = np.array([0, 0.])
		self.cusp_points[LOWER_CUSP] = np.array([1, 0.])
		self.cusp_points[OUTLET_CUSP]= np.array([1, 1.])
		self.cusp_points[UPPER_CUSP] = np.array([0, 1.])
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.scatter(self.nodes[:, 0], self.nodes[:, 1])
		ax.scatter(np.array([self.cusp_points[i][0] for i in self.cusp_points]), np.array([self.cusp_points[i][1] for i in self.cusp_points]))
		plt.show()
		self.cusp_lists = {}
		for i in [INLET_CUSP, LOWER_CUSP, OUTLET_CUSP, UPPER_CUSP]:
			self.cusp_lists[i] = np.array([u - self.cusp_points[i] for u in self.nodes])
			for u in self.cusp_lists[i]:
				wrap_array = np.array([1, -1.0/self.twist_num])
				if lin.norm((u - wrap_array)) < lin.norm(u):
					u -= wrap_array
				elif lin.norm((u + wrap_array)) < lin.norm(u):
					u += wrap_array
			self.cusp_map(self.cusp_lists[i], i)
			self.cusp_lists[i] = np.array([u for u in self.cusp_lists[i]])
		self.init_stencil(stencil_size)

	def get_domain(self, node):
		y, x = node[1], node[0]
		if y < .5/self.twist_num:
			if x < .5/self.twist_num:
				return INLET_CUSP
			if x > (1 - .5/self.twist_num):
				print('case 1')
				return LOWER_CUSP
			return LOWER_BOUND
		if y > (1 - .5/self.twist_num):
			if x > (1 - .5/self.twist_num):
				return OUTLET_CUSP
			if x < .5/self.twist_num:
				return UPPER_CUSP
			return UPPER_BOUND
		if y < 1.5/self.twist_num:
			if x < .5/self.twist_num:
				print('case 2')
				return LOWER_CUSP
		if y > (1.0 - 1.5/self.twist_num):
			if x > (1 - .5/self.twist_num):
				return UPPER_CUSP
		return INTERIOR

	def transform_nodes(self, node_list, domain):
		new_list, wrap_list = self.set_center(node_list, domain)
		ref_vec = None
		if domain in [INLET_CUSP, UPPER_CUSP, OUTLET_CUSP, LOWER_CUSP]:
			ref_vec = self.cusp_map(new_list, domain)
		return new_list, ref_vec, wrap_list

	def set_center(self, node_list, domain):
		if domain in [UPPER_BOUND, LOWER_BOUND]:
			center = np.array([node_list[0][0], float(domain == UPPER_BOUND)])
		elif domain == INTERIOR:
			center = node_list[0]
		else:
			center = self.cusp_points[domain]
		new_list = np.array([u - center for u in node_list])
		wrap_list = self.wrap_nodes(new_list)
		return new_list, wrap_list

	def wrap_nodes(self, node_list):
		wrap_list = np.array([0.]*len(node_list))
		if node_list[0][0] < -.5:
			wrap_list -= 1
		if node_list[0][0] > .5:
			wrap_list += 1
		wrap_vec = np.array([2*np.pi, -2*np.pi*self.lam])
		for i, u in enumerate(node_list):
			if u[0] < -.5:
				wrap_list[i] += 1.
				node_list[i][0] += 1
				node_list[i][1] -= 1.0/self.twist_num
			if u[0] > .5:
				wrap_list[i] -= 1.
				node_list[i][0] -= 1
				node_list[i][1] += 1./self.twist_num
		return wrap_list

	def cusp_map(self, node_list, domain):
		if domain in [LOWER_CUSP, OUTLET_CUSP]:
			i = 0
		else:
			i = 1

		new_list = []
		for u in node_list:
			p = u[0] + u[1]*1j
			q = u[0] + u[1]*1j
			q = 1./(1 - q) - 1./(q)
			new_list.append(np.array([q.real, q.imag]))
			p = np.sqrt(p)
			u[0], u[1] = p.real, p.imag
			u *= np.sign(u[i])

		new_list = np.array(new_list)
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.scatter(new_list[:, 0], new_list[:, 1])
		plt.show()		
		u = node_list[0]
		reflection = np.eye(2)
		reflection[i, i] = -1
		new_transform = np.array([[2*u[0], 2*u[1]], [-2*u[1], 2*u[0]]])
		node_list[...] = np.array([new_transform @ u for u in node_list])
		new_transform /= lin.norm(new_transform, axis=1, keepdims=True)
		ref =  new_transform @ reflection @ lin.inv(new_transform)
		ref_vec = np.zeros(2)
		ref_vec[i] = 1.
		return new_transform @ ref_vec

	def init_stencil(self, p_num):
		NN_tree = KDTree(self.nodes_mod)
		NN_dist, NN_list = NN_tree.query(self.nodes_mod, k=p_num)
		NN_cusp_list = {}
		for i in [INLET_CUSP, OUTLET_CUSP, LOWER_CUSP, UPPER_CUSP]:
			cusp_tree = KDTree(self.cusp_lists[i])
			NN_cusp_list[i] = cusp_tree.query(self.cusp_lists[i], k=p_num)[1]
		row =  [[] for i in range(4)]
		col =  [[] for i in range(4)]
		data = [[] for i in range(4)]
		#def h_deriv(p_list, metric, i, group, bc):
		#	c_num = 1
		#	factor = 1
		#	while c_num > 1e-6:
		#		if i == 2:
		#			c_num, weights = hermite_laplace(p_list, factor*metric, group=group, bc=bc)
		#		else:
		#			c_num, weights = hermite_deriv(p_list, factor*metric, i, group=group, bc=bc)
		#		factor *= 2
		#	print(factor, c_num)
		#	return weights
		for i, NN in enumerate(NN_list):

			domain = self.get_domain(self.nodes[i])
			if domain in [INLET_CUSP, OUTLET_CUSP, LOWER_CUSP, UPPER_CUSP]:
				p_list = np.array([self.nodes[j] for j in NN_cusp_list[domain][i]])
				print(p_list.shape)
			else:
				p_list = np.array([self.nodes[j] for j in NN_list[i]])
				print(p_list.shape)
			if domain in [UPPER_BOUND, UPPER_CUSP, OUTLET_CUSP]:
				bc = np.array([[1.0]*p_num])
			else:
				bc = None
			print(domain)
			p_list, ref_vec, wrap_list = self.transform_nodes(p_list, domain)
			p_list = np.array(p_list)
#			p_range = np.max(p_list, axis=0) - np.min(p_list, axis=0)
#			eps = .1
#			metric = np.diag(eps/p_range**2)

			weights = self.phs(np.array(p_list), ref_vec, bc=bc)
			weights[0] /= 2*np.pi
			weights[2] /= 4*np.pi**2
			weights[1] /= self.width
			weights[3] /= self.width**2
			wrap_weights = np.array(weights)[:, :p_num] @ wrap_list
			for j in range(4):
				row[j].extend([i]*p_num)
#				row[j + 3].extend([i]*(p_num))
				col[j].extend(NN)
#				col[j + 3].extend(NN)
				data[j].extend(weights[j][:p_num])
#				data[j + 3].append(1.)
#				data[j + 3].extend(-weights[j][p_num:])
				if wrap_weights[j] != 0:
					row[j].append(i)
					col[j].append(self.nodes.shape[0])
					data[j].append(wrap_weights[j])

		self.deriv_stencils = [spar.csr_matrix(spar.coo_matrix( (data[j], (row[j], col[j])) )) for j in range(4)]
#		self.hermite_stencils = [spar.csr_matrix(spar.coo_matrix((data[j], (row[j], col[j])) )) for j in range(3, 6)]

	def differentiate(self, index, U, target):
		target[...] = self.deriv_stencils[index] @ U
#		target[...] = spsolve(self.hermite_stencils[index], x)

	def phs(self, nodes, ref_vec, bc=None):
		# Find stencil for first derivatives and homogeneous second derivatives using rbf phs
		# GA_coords are the coordinates relative to the central node. s is the s coordinate of the central node
		# edge_list is a _dictionary_ with keys corresponding to the nodes which are on an edge

		REF_FLAG = np.any(np.abs(nodes[0]) > 1e-6)
		if bc is None:
			bc = []
		bc_num = len(bc)
		nodes = np.array([u for u in nodes])

		if REF_FLAG:
			ref_vec = ref_vec if ref_vec is not None else np.array([0., 1.])
			ref_vec /= lin.norm(ref_vec)
			perp_vec = [-ref_vec[1], ref_vec[0]]
			rotation = np.array([[ref_vec[1], ref_vec[0]], [-ref_vec[0], ref_vec[1]]])
			reflection = np.outer(perp_vec, perp_vec) - np.outer(ref_vec, ref_vec)
		p_num = self.evaluator.term_num if not REF_FLAG else self.evaluator.term_num_ref

		eval_mat = np.zeros((nodes.shape[0] + p_num + bc_num, nodes.shape[0] + p_num + bc_num))

		deriv_mat = np.zeros((nodes.shape[0] + p_num + bc_num, 4))

		eval_mat[:nodes.shape[0], :nodes.shape[0]] = np.array([[lin.norm(u - v)**3 for u in nodes] for v in nodes])
		if REF_FLAG:
			eval_mat[:nodes.shape[0], :nodes.shape[0]] += np.array([[-lin.norm(u - reflection @ v)**3 for u in nodes] for v in nodes])

		i_finder = np.zeros(2)

		terms = self.evaluator.terms if not REF_FLAG else self.evaluator.terms_ref
		for i in range(2):
			u = nodes[0]
			# First derivatives and homogeneous second derivatives
			if not REF_FLAG:
				i_finder[...] = 0
				i_finder[i] = 1
				index = terms.index(tuple(i_finder))

				deriv_mat[nodes.shape[0] + index, i] = 1

				i_finder[...] = 0
				i_finder[i] = 2
				index = terms.index(tuple(i_finder))
				deriv_mat[nodes.shape[0] + index, i + 2] = 2

			deriv_mat[:nodes.shape[0], i] = np.array([3*((u - v))[i]*lin.norm((u - v)) for v in nodes])
			if REF_FLAG:
				deriv_mat[:nodes.shape[0], i] += np.array([-3*((u - reflection @ v))[i]*lin.norm((u - reflection @ v)) for v in nodes])
			deriv_mat[:nodes.shape[0], i + 2] = np.array([ 3*lin.norm(u - v) + 3*((u - v)[i]**2)/lin.norm(u - v) if lin.norm(u - v) != 0. else 0. for v in nodes])

			if REF_FLAG:
				deriv_mat[:nodes.shape[0], i + 2] += np.array([-3*lin.norm(u - reflection @ v) - 3*((u - reflection @ v)[i]**2)/lin.norm(u - reflection @ v) if lin.norm(u - reflection @ v) != 0. else 0. for v in nodes])

		if REF_FLAG:
			deriv_mat[nodes.shape[0]:(-bc_num if bc_num else None)] = self.evaluator.evaluate_mono_derivs_ref(nodes[0], rotation).T

		for i in range(nodes.shape[0]):
			poly_list = self.evaluator.evaluate_monomials(nodes[i]) if not REF_FLAG else self.evaluator.evaluate_monomials_ref(nodes[i], rotation)
			eval_mat[i, nodes.shape[0]:(-bc_num if bc_num else None)] = poly_list[...]
			eval_mat[nodes.shape[0]:(-bc_num if bc_num else None), i] = poly_list[...]

		for i, u in enumerate(bc):
			eval_mat[-(i + 1), :nodes.shape[0]] = u
			eval_mat[:nodes.shape[0], -(i + 1)] = u
		try:
			weights = lin.solve(eval_mat, deriv_mat)
		except:
			print(REF_FLAG)
			raise ValueError
		print('max_error: ', np.max(np.abs(np.dot(eval_mat, weights) - deriv_mat)))
		weights = weights[:nodes.shape[0]].T

		return weights

class QuenchSim:
	# Class representing quenching on a coiled annulus superconductor
	def __init__(self, r_1, r_2, twist_num, h, stencil_size, degree, **kwargs):
		# initialize simulation
		if 'simple' in kwargs:
			if kwargs['simple']:
				self.space = NodeFinderSimple(r_1, r_2, h)
				self.SIMPLE = True
			else:
				self.space = NodeStepper(twist_num, r_1, r_2, h)
				self.SIMPLE = False
		else:
			self.space = NodeStepper(twist_num, r_1, r_2, h)
			self.SIMPLE = False
		self.stencil = QuenchHermite(self.space, stencil_size)
		output = self.space.get_spatial_data()
		self.nodes = np.array(output['nodes'])
		self.p_num = self.nodes.shape[0]
		self.lam = output['lambda']
		if self.SIMPLE:
			self.lam = (r_2 - r_1)/(np.pi)
		self.r = self.nodes[:, 1] + self.nodes[:, 0]*self.lam
		#self.outlet_num = output['outlet_num']
		#self.dirichlet_num = output['dirichlet_num']
		#self.pair_num = self.dirichlet_num + output['pair_offset']
		#self.pair_offset = output['pair_offset']

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

		for i in range(3):
			self.stencil.differentiate(i, self.U_val, self.U_deriv[i])


	def get_U_t(self):
		self.get_derivs()
		self.U_t[...] = 0.0
		# Add up the individual terms. We use the product rule for derivatives.
		self.U_t += self.U_deriv[3]*self.eta[1]
		#self.U_t += self.U_deriv[1]*self.eta[1]/self.r
#		self.U_t += self.eta_deriv[1]*self.U_deriv[1]
		self.U_t += self.U_deriv[2]*self.eta[0]
#		self.U_t += self.U_deriv[0]*self.eta_deriv[0]/self.r**2
		#self.U_t -= 2*self.U_deriv[0]*self.eta[0]*self.lam/self.r**3


	def set_U(self, func):
		self.U_val[...] = 0
		self.U_val[:self.p_num] = np.array([func(u) for u in self.nodes])
		#self.U_val[:self.outlet_num] = self.outlet_slope
		if not self.SIMPLE:
			self.U_val[-1] = func(np.array([0.0, self.r_1 + 2*np.pi*self.lam])) - func([2*np.pi, self.r_1])


	def solve_U_t(self):

		def objective(U):
			self.U_val[1:self.p_num] = U[...]
			self.U_val = 0.0
			self.get_U_t()
			return self.U_t[(1):].copy()

		return root(objective, self.U_val[(1):self.p_num].copy())


	def step_forward(self, t_step, step_num):
		rk_storage = np.zeros((4, self.U_t.shape[0]))

		for i in range(step_num):
			print(i)
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

