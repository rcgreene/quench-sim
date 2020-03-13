import numpy as np
import numpy.linalg as lin
import scipy as sp
import scipy.linalg as lin
import scipy.sparse as spar
from numba import jit
from scipy.optimize import minimize
from scipy.spatial import KDTree
from numpy.random import rand

def hexagonal_grid(twist_num, subdivision):
	lattice_vectors = np.array([
								[.5/(twist_num*subdivision), .5/(twist_num*subdivision)],
								[1./(twist_num*subdivision), 0.]])
	nodes = []
	for i in range(0, 2*twist_num*subdivision):
		if i < twist_num*subdivision:
			a = -(i//2)
			b = twist_num*subdivision - (i//2)
		elif i == twist_num*subdivision:
			a = -((i - 1)//2)
			b = twist_num*subdivision - ((i - 1)//2)
		else:
			a = -((i - 1)//2)
			b = twist_num*subdivision - ((i - 1)//2)
		for j in range(a, b):
			p = i*lattice_vectors[0] + j*lattice_vectors[1]
			p[1] += .25/(twist_num*subdivision)
			nodes.append(p)
	nodes = np.array(nodes)
	print('angles:', 1 - np.max(nodes[:, 0]), np.min(nodes[:, 0]))
	print('radii:', np.min(nodes[:, 1]), 1 - np.max(nodes[:, 1]))
	return nodes

class NodeFinder:
	def __init__(self, r_1, r_2, twist_num, h):
		
		self.r_1 = r_1
		self.r_2 = r_2
		self.del_r = (r_2 - r_1)/twist_num
		self.lam = self.del_r/(2*np.pi)     # Using the notation from the paper; not lambda because it's a keyword
		self.h = h
		self.twist_num = twist_num
		subdivision = 4 # for testing
		self.nodes=[]
		self.init_cusp_regions(subdivision)
		self.init_interior_regions(subdivision)
		#self.init_edge_regions(subdivision)
		self.rescale_nodes()
		#self.init_boundary_nodes()
		#self.init_interior_nodes()

	def init_interior_regions(self, subdivision):
		nodes = hexagonal_grid(self.twist_num, subdivision)
		interior_nodes = []
		for p in nodes:
			if (p[1] <= 0) or (p[1] >= 1):
				# p is not in the final node list in this case
				continue
#			elif (lin.norm(p - np.array([1.0/self.twist_num, 0])) < .5/self.twist_num):
				# p is not in the final node list in this case
#				continue
			interior_nodes.append(p)
		self.add_nodes(interior_nodes, subdivision)
		self.interior_index = len(self.nodes)

	def init_cusp_regions(self, subdivision):
		cusp_nodes = []
		corner_nodes = []
		lat_vecs = np.array([
			[.5,-.5],
			[0.,1.]])/(2*subdivision)

		for i in range(4*subdivision):
			for j in range(4*subdivision):
				p = (i + .5)*lat_vecs[0] + j*lat_vecs[1]
				if lin.norm(p) < 1:
					cusp_nodes.append(p[0] + 1j*p[1])
					if 2*j <= i:
						corner_nodes.append(cusp_nodes[-1].copy())

		for i in range(len(cusp_nodes)):
			# Transform by squaring in the complex plane
			p = cusp_nodes[i]
			p = .5 * (p**2) / self.twist_num
			p_bot = np.array([p.real, p.imag + 1.0/self.twist_num])
			if p_bot[0] < 0:
				p_bot += np.array([1., -1.0/self.twist_num])
			p_top = np.array([1 - p.real, 1 - 1.0/self.twist_num - p.imag])
			if p_top[0] > 1.:
				p_top += np.array([-1., 1.0/self.twist_num])
			cusp_nodes[i] = p_top
			cusp_nodes.append(p_bot)

		for i in range(len(corner_nodes)):
			p = corner_nodes[i]
			p = .5 * (p**2)/self.twist_num
			p_bot = np.array([p.real, -p.imag])
			p_top = np.array([1 - p.real, 1 + p.imag])
			cusp_nodes.append(p_bot)
			cusp_nodes.append(p_top)
		self.cusp_index = self.add_nodes(cusp_nodes, subdivision)

	def init_edge_regions(self, subdivision):
		lat_vecs = np.array([
						[1.,0.],
						[-.5,.5]
			]) / (2*subdivision*self.twist_num)
		edge_nodes = []

		for i in range(3*self.twist_num*subdivision):
			for j in range(3*subdivision):
				p = i*lat_vecs[0] + (j + .5)*lat_vecs[1]
				if (p[0] < 0.) or (p[0] > 1.):
					continue
				p[1] *= 1 - np.cos(np.pi*min(p[1]*self.twist_num, 1.))
				edge_nodes.append(p)
				v = np.array([1 - p[0], 1 - p[1]])
				edge_nodes.append(v)
		self.edge_index = self.add_nodes(edge_nodes, subdivision)

	def add_nodes(self, node_list, subdivision):
		# Routine for integrating nodes from different domains to the main node set.
		if not self.nodes:
			self.nodes = node_list
			return len(self.nodes)
		dist_checker = KDTree(self.nodes)
		dist_list = dist_checker.query(node_list, k=1)[0]
		for i, p in enumerate(node_list):
			if dist_list[i] > .25/(self.twist_num*subdivision):
				self.nodes.append(p)
		return len(self.nodes)

	def rescale_nodes(self):
		# rescale nodes to a rectangular domain
		self.nodes = np.array(self.nodes)
		self.nodes[:, 1] *= self.r_2 - self.r_1
		self.nodes[:, 1] += self.r_1
		self.nodes[:, 0] *= 2*np.pi

	def get_node_info(self):
		output = {}
		output['nodes'] = self.nodes
		output['dirichlet_num'] = self.dirichlet_num
		output['outlet_num'] = self.outlet_num
		output['pair_offset'] = self.pair_offset
		output['boundary_num'] = self.boundary_num
		output['lambda'] = self.lam
		output['width'] = self.r_2 - self.r_1
		return output

	def init_boundary_nodes(self):
		#Initialize inlet and outlet nodes--dirichlet boundary conditions
		self.nodes = []
#		n = np.ceil(2*self.del_r / self.h)       #Boundary nodes are twice as densely packed as interior nodes
#		a, b = self.r_1, self.r_1 + self.del_r
#		self.nodes = [[0.0, s] for s in np.linspace(a, b, n)]		

#		a, b = self.r_2 - 2*self.del_r, self.r_2 - self.del_r
#		self.nodes.extend([ [2*np.pi, s] for s in np.linspace(a, b, n)])

		self.outlet_num = len(self.nodes)

		#Initialize top and bottom nodes--dirichlet boundary conditions
		n = int(np.ceil(2*(self.r_2 - self.r_1 - self.del_r)/self.h))
		a, b = 0.0, 2*np.pi
		self.nodes.extend([ [theta, self.r_1] for theta in np.linspace(a, b, n)])

		a, b = 0.0, 2*np.pi
		self.nodes.extend([ [theta, self.r_2 - self.del_r] for theta in np.linspace(a, b, n)])		

		self.dirichlet_num = len(self.nodes)

		#Initialize wrap-around nodes, these nodes are paired. Pairs must always have the same partial derivatives.
		#n = int(np.ceil(2*(self.r_2 - self.r_1 - 2*self.del_r)/self.h))
		#self.pair_offset = n - 2
		self.pair_offset = 0
		#left pairs
		#a, b = self.r_1 + self.del_r, self.r_2 - self.del_r
		#self.nodes.extend([ [0.0, s] for s in np.linspace(a, b, n)[1:-1]])
		#right pairs
		#a, b = self.r_1, self.r_2 - 2*self.del_r
		#self.nodes.extend([ [2*np.pi, s] for s in np.linspace(a, b, n)[1:-1]])

		self.boundary_num = len(self.nodes)
		self.pair_num = int(self.boundary_num - self.dirichlet_num)

	def init_interior_nodes(self):

		#Initialize hexagonal lattice
		t_1, t_2 = np.pi/12, 5*np.pi/12
		e_1, e_2 = np.array( [np.cos(t_1), np.sin(t_1)])*self.h/(self.r_2 - self.r_1), np.array([np.cos(t_2), np.sin(t_2)])*self.h/(self.r_2 - self.r_1)
		k_num = int(np.ceil(3.0/e_1[0]))

		#insert interior nodes
		interior_nodes = []
		for i in range(-k_num, k_num):
			for j in range(-k_num, k_num):
				x = i*e_1 + j*e_2
				frac = 0*self.twist_num
				if x[1] > 1 - frac:
					x[1] = 1 - frac + (x[1] + frac - 1)/20
				if x[1] < frac:
					x[1] =  frac - (frac - x[1])/10
				if np.all(np.abs(x - .5) < .5):
					x = -np.cos(x*np.pi)/2 + .5
					x[1] = x[1]*(self.r_2 - self.r_1 - self.del_r) + self.r_1
					x[0] = 2*np.pi*x[0]
					self.nodes.append(x)
		self.nodes = np.array(self.nodes)

class NodeFinderSimple(NodeFinder):
	# node finder for the special case of a domain whose inlet/outlet covers the entire left and right boundaries
	def __init__(self, r_1, r_2, h):
		NodeFinder.__init__(self, r_1, r_2, 1e999, h)
		self.nodes[:,0] *= .5

	def init_boundary_nodes(self):
		self.del_r = 0
		n = int(np.ceil(2*(self.r_2 - self.r_1)/self.h))
		self.nodes = [[0.0, s] for s in np.linspace(self.r_1, self.r_2, n)[1:-1]]
		self.nodes.extend([2*np.pi, s] for s in np.linspace(self.r_1, self.r_2, n)[1:-1])
		self.outlet_num = len(self.nodes)

		self.nodes.extend([alpha, self.r_1] for alpha in np.linspace(0, 2*np.pi, n))
		self.nodes.extend([alpha, self.r_2] for alpha in np.linspace(0, 2*np.pi, n))

		self.dirichlet_num = len(self.nodes)

		self.pair_offset = 0
		self.pair_num = 0
		self.boundary_num = self.dirichlet_num

class NodeStepper:
	def __init__(self, twist_num, r_1, r_2, h):
		self.twist_num = twist_num
		self.h = h
		self.d_num = int(1.//h + 1) # might be better choices
		self.r_1 = r_1
		self.r_2 = r_2
		self.pdp_dict = {(i, j) : [] for i, j in np.ndindex((self.d_num, self.d_num))}
		pdp_list = []
		self.node_list = []
		self.angle_list = []
		self.make_radius_func()
		self.seed_nodes()
		self.fill_domain()
		self.remove_edge_nodes()
		self.rescale()

	def get_spatial_data(self):
		output = {}
		output['nodes']  = self.node_list
		output['lambda'] = (self.r_2 - self.r_1)/(2*np.pi*self.twist_num)
		output['r_1']    = self.r_1
		output['r_2']    = self.r_2
		output['twist_num'] = self.twist_num
		return output

	def seed_nodes(self):
		A = rand(100)
		for i, x in enumerate(np.linspace(0,1, 100)[1:-1]):

			self.add_pdp(np.array([x, 0.001 + .001*A[i]]))
		for x in np.linspace(0, 1, 7)[1:-1]:
			self.add_pdp(np.array([.001, x]))
			self.add_pdp(np.array([.999, x]))

	def fill_domain(self):
		while 1:
			next_node = self.select_next()
			if next_node is None:
				break
			self.node_list.append(next_node)
			angles = self.remove_close_pdps(next_node)
			if angles[1] is None:
				next_node = self.select_next()
				if next_node is not None:
					self.node_list.append(next_node)
				break
			self.fill_arc(angles, next_node)

	def rescale(self):
		for p in self.node_list:
			p[0] *= 2*np.pi
			p[1] = p[1]*(self.r_2 - self.r_1) + self.r_1

	def remove_edge_nodes(self):
		new_list = []
		for p in self.node_list:
			r = self.radius_func(p)
			if p[1] > .5:
				if (min(1 - p[0], 1 - p[1]) > .5*r):
					new_list.append(p)
			else:
				if (min(p[0], p[1]) > .5*r):
					new_list.append(p)

		self.node_list = new_list

	def add_pdp(self, node):
		i, j = (int(node[i]*self.d_num) for i in range(2))
		self.pdp_dict[(i, j)].append(node)

	def remove_close_pdps(self, node):
		i, j = (int(node[i]*self.d_num) for i in range(2))
		r = self.radius_func(node)
		closest_nodes, close_dist, angles = [None, None], [1e999, 1e999], [np.pi, 0]
		for k in range(self.d_num):
			for l in range(self.d_num):
				dom = (k, l)
				pdp_list = self.pdp_dict[dom]
				n = 0
				while n < len(pdp_list):
					dist = lin.norm(pdp_list[n] - node)
					if dist < r:
						pdp_list.pop(n)
					else:
						sep = pdp_list[n] - node
						angle = np.arctan2(sep[1], sep[0])
						side = np.abs(angle) < np.pi/2
						if dist < close_dist[side]:
							if angles[side] < -np.pi/2:
								angles[side] += 2*np.pi
							closest_nodes[side] = (dom, n)
							close_dist[side] = dist
						n += 1
		print('dist', close_dist)
		return angles

	def fill_arc(self, angles, node):
		r = self.radius_func(node)
		theta_list = np.linspace(*angles, 9)[1:-1]
		print(angles)
		for alpha in theta_list:
			next_pdp = node + np.array([r*np.cos(alpha), r*np.sin(alpha)])
			if np.all(np.abs(next_pdp - .5) < .5):
				self.add_pdp(next_pdp)

	def select_next(self):

		next_node = None

		for j in range(self.d_num):
			for i in range(self.d_num):
				for n, p in enumerate(self.pdp_dict[(i, j)]):
					if next_node is None:
						next_node = p
					elif p[1] < next_node[1]:
						next_node = p
		print(next_node)
		return next_node

	def make_radius_func(self):
		twist_num = self.twist_num
		h = self.h
		@jit
		def edge_distance(u):
			if u[1] < 1.0/twist_num:
				d = u[0]
			elif u[1] > 1 - 1.0/twist_num:
				d = 1 - u[0]
			else:
				d = np.sqrt(min((u[0]**2 + (u[1] - 1.0/twist_num)**2), ((1 - u[0])**2 + (u[1] - 1 + 1.0/twist_num)**2)))
			d = min(d, u[1])
			d = min(d, 1. - u[1])
			return d

		@jit
		def radius_func(u):
			d = edge_distance(u)
			d = min(1., d*2)
			return h*max(np.sqrt(np.sin(np.pi*d/2)), 2e-1)
		self.radius_func = radius_func