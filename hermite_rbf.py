import numpy as np
import numpy.linalg as lin
from scipy.linalg import solve
from numba import jit
from numba import cuda

def hexagonal_shell(d):
	lattice_vecs = np.array([[.5, np.sqrt(3)/2],[1, 0.]])
	nodes = [d*lattice_vecs[0,:]]
	for i in range(d):
		nodes.append(nodes[-1] - lattice_vecs[0] + lattice_vecs[1])
	for i in range(d):
		nodes.append(nodes[-1] - lattice_vecs[0])
	for i in range(d):
		nodes.append(nodes[-1] - lattice_vecs[1])
	for i in range(d):
		nodes.append(nodes[-1] + lattice_vecs[0] - lattice_vecs[1])
	for i in range(d):
		nodes.append(nodes[-1] + lattice_vecs[0])
	for i in range(d - 1):
		nodes.append(nodes[-1] + lattice_vecs[1])
	return nodes

def hexagon_cell(depth):
	
	nodes = [np.array([0,0])]
	for i in range(1, depth + 1):
		nodes.extend(hexagonal_shell(i))
	return nodes

def hermite_hexagonal(depth, eps):
	nodes = hexagon_cell(depth)
	return nodes, hermite_laplace(nodes, eps), hermite_deriv(nodes, eps, 0), hermite_deriv(nodes, eps, 1)

def get_node_separation(nodes, group, metric):

	separation_list = []
	dist_mat_list = []
	rbf_mat_list = []
	p_num = len(nodes)
	met_norm = lambda u: u @ metric @ u
	for c, g in group: 
		separation_list.append([[u - g @ v for v in nodes] for u in nodes])
		dist_mat_list.append(np.array([[met_norm(u) for u in vec_row] for vec_row in separation_list[-1]]))
		rbf_mat_list.append(np.array([[c*np.exp(-dist_mat_list[-1][i,j]) for i in range(p_num)] for j in range(p_num)]))
	return separation_list, dist_mat_list, rbf_mat_list

def hermite_laplace(nodes, metric, group=None, bc=None):

	p_num = len(nodes)
	deriv_num = len(nodes)*2 - 1
	if bc is None:
		bc = [np.ones(p_num)]
	if group is None:
		group = [(1, np.eye(2))]
	
	met_norm = lambda u: u @ metric @ u
	s_list, dist_list, rbf_list = get_node_separation(nodes, group, metric)
	eval_mat = np.zeros((p_num*2 - 1 + len(bc), p_num*2 - 1 + len(bc)))
	for i, cond in enumerate(bc):
		eval_mat[-i - 1, :p_num] = cond
		eval_mat[:p_num, -i - 1] = cond

	rhs_mat = np.zeros(2*p_num - 1 + len(bc))
	for seps, dists, rbfs in zip(s_list, dist_list, rbf_list):
		eval_mat[:p_num, :p_num] += np.array([[rbfs[i, j] for i in range(p_num)] for j in range(p_num)])
		eval_mat[p_num:deriv_num, :p_num] += np.array([[(-2*np.sum(metric) + 4*lin.norm(metric @ seps[i][j])**2)*rbfs[i,j] for i in range(p_num)] for j in range(1, p_num)])
		eval_mat[:p_num, p_num:deriv_num] = eval_mat[p_num:deriv_num, :p_num].T
		eval_mat[p_num:deriv_num, p_num:deriv_num] += np.array([[(((-2*np.sum(metric) + 4*lin.norm(metric @ seps[i][j])**2)**2) + 8*np.sum(metric)**2 - seps[i][j] @ metric**3 @ seps[i][j])*rbfs[i,j] for i in range(1, p_num)] for j in range(1, p_num)])
		rhs_mat[:p_num] += np.array([(-2*np.sum(metric) + 4*lin.norm(metric @ seps[0][i])**2)*rbfs[0, i] for i in range(p_num)])
		rhs_mat[p_num:deriv_num] += np.array([rbfs[0, i]*(((-2*np.sum(metric) + 4*lin.norm(metric @ seps[0][i])**2)**2) + 8*np.sum(metric)**2 - seps[0][i] @ metric**3 @ seps[0][i]) for i in range(1, p_num)])

	weights = lin.solve(eval_mat, rhs_mat)
	c_num = (np.max(np.abs(np.dot(eval_mat, weights) - rhs_mat)))
	return c_num, weights[:deriv_num]

def hermite_deriv(nodes, metric, axis, group=None, bc=None):

	p_num = len(nodes)
	deriv_num = len(nodes)*2 - 1
	if bc is None:
		bc = [np.ones(p_num)]
	if group is None:
		group = [(1, np.eye(2))]
	
	s_list, dist_list, rbf_list = get_node_separation(nodes, group, metric)
	eval_mat = np.zeros((p_num*2 - 1 + len(bc), p_num*2 - 1 + len(bc)))
	for i, cond in enumerate(bc):
		eval_mat[-i - 1, :p_num] = cond
		eval_mat[:p_num, -i - 1] = cond

	rhs_mat = np.zeros(2*p_num - 1 + len(bc))
	for seps, dists, rbfs in zip(s_list, dist_list, rbf_list):
		eval_mat[:p_num, :p_num] += np.array([[rbfs[i, j] for i in range(p_num)] for j in range(p_num)])
		eval_mat[p_num:deriv_num, :p_num] += np.array([[-2*(metric @ seps[i][j])[axis]*rbfs[i,j] for i in range(p_num)] for j in range(1, p_num)])
		eval_mat[:p_num, p_num:deriv_num] = eval_mat[p_num:deriv_num, :p_num].T
		eval_mat[p_num:deriv_num, p_num:deriv_num] += np.array([[-rbfs[i,j]*(4*(metric @ seps[i][j])[axis]**2 - 2*metric[axis, axis]) for i in range(1, p_num)] for j in range(1, p_num)])
		rhs_mat[:p_num] += np.array([-2*(metric @ seps[0][i])[axis]*rbfs[0, i] for i in range(p_num)])
		rhs_mat[p_num:deriv_num] += np.array([rbfs[0, i]*(-2*metric[axis, axis] + 4*(metric @ seps[0][i])[axis]**2) for i in range(1, p_num)])

	weights = lin.solve(eval_mat, rhs_mat)

	c_num = (np.max(np.abs(np.dot(eval_mat, weights) - rhs_mat)))
	return c_num, weights[:deriv_num]


def test(depth, eps):

	return hermite_hexagonal(depth, eps)