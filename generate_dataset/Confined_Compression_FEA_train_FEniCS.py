##########################################################################################
# import necessary modules (list of all simulation running modules)
##########################################################################################
import matplotlib.pyplot as plt
from dolfin import *
import os
import sys
import numpy as np
from mshr import *
from scipy import interpolate
#from ufl import *
##########################################################################################
# input text files
##########################################################################################
num = int(sys.argv[1])

mref = 5  #int(sys.argv[2])

flag_quad = 2 # int(sys.argv[3]) # --> if flag_quad == 1, 

fname = 'train_data_' + str(num)

input_folder = 'input_data' # location of the bitmaps

data_import = np.loadtxt(input_folder + '/' + fname + '.txt') # <--MNIST data 
data = np.zeros(data_import.shape)
for jj in range(0,data.shape[0]):
	for kk in range(0,data.shape[1]):
		data[jj,kk] = data_import[int(27.0 - kk),jj] #jj is columns of input, kk is rows
		
folder_name =  'folder' + '_' + fname + '_' + 'mesh' + str(mref) + '_quad' + str(flag_quad)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
##########################################################################################
##########################################################################################
# compliler settings / optimization options 
##########################################################################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = flag_quad
##########################################################################################

################  ~ * ~ * ~ * ~ |
################  ~ * ~ * ~ * ~ |	--> before solver loop 
################  ~ * ~ * ~ * ~ |

##########################################################################################
# mesh geometry  
##########################################################################################
p_1_x = 0; p_1_y = 0;
p_2_x = 28.0; p_2_y = 28.0; 
rect = Rectangle(Point(p_1_x,p_1_y),Point(p_2_x,p_2_y))
mesh = generate_mesh(rect,int(28*mref))
##########################################################################################
# mesh and material prop
##########################################################################################
P2 = VectorElement("Lagrange", mesh.ufl_cell(), flag_quad)
TH = P2
W = FunctionSpace(mesh, TH)
V = FunctionSpace(mesh, 'CG', 1)
back = 1.0
high = 100.0 
nu = 0.3
material_parameters = {'back':back, 'high':high, 'nu':nu}

def bitmap(x,y): #there could be a much better way to do this, but this is working within the confines of ufl
	total = 0   
	for jj in range(0,data.shape[0]):
		for kk in range(0,data.shape[1]):
			const1 = conditional(x>=jj,1,0) # x is rows
			const2 = conditional(x<jj+1,1,0) 
			const3 = conditional(y>=kk,1,0) # y is columns 
			const4 = conditional(y<kk+1,1,0) #less than or equal to? 
			sum = const1 + const2 + const3 + const4
			const = conditional(sum>3,1,0) #ufl equality is not working, would like to make it sum == 4 
			total += const*data[jj,kk]
	return total

class GetMat:
	def __init__(self,material_parameters,mesh):
		mp = material_parameters
		self.mesh = mesh
		self.back = mp['back']
		self.high = mp['high']
		self.nu = mp['nu']
	def getFunctionMaterials(self, V):
		self.x = SpatialCoordinate(self.mesh)
		val = bitmap(self.x[0],self.x[1])
		E = val/255.0*(self.high-self.back) + self.back
		effectiveMdata = {'E':E, 'nu':self.nu}
		return effectiveMdata

mat = GetMat(material_parameters, mesh)
EmatData = mat.getFunctionMaterials(V)
E  = EmatData['E']
nu = EmatData['nu']
lmbda, mu = (E*nu/((1.0 + nu )*(1.0-2.0*nu))) , (E/(2*(1+nu)))
matdomain = MeshFunction('size_t',mesh,mesh.topology().dim())
dx = Measure('dx',domain=mesh, subdomain_data=matdomain)
##########################################################################################
# apply traction, and body forces (some boundary conditions are within the solver b/c they update)
##########################################################################################
T  = Constant((0.0, 0.0))  # Traction force on the boundary
B  = Constant((0.0, 0.0))

top  =  CompiledSubDomain("near(x[1], topCoord)", topCoord = p_2_y)
btm  =  CompiledSubDomain("near(x[1], btmCoord)", btmCoord = p_1_y)
lft  =  CompiledSubDomain("near(x[0], lftCoord)", lftCoord = p_1_x)
rgt  =  CompiledSubDomain("near(x[0], rgtCoord)", rgtCoord = p_2_x)

btmBC = DirichletBC(W.sub(1), Constant((0.0)), btm)
lftBC = DirichletBC(W.sub(0), Constant((0.0)), lft)
rgtBC = DirichletBC(W.sub(0), Constant((0.0)), rgt)

##########################################################################################
# define finite element problem
##########################################################################################
u = Function(W)
du = TrialFunction(W)
v = TestFunction(W)
##########################################################################################
##########################################################################################

################  ~ * ~ * ~ * ~ |
################  ~ * ~ * ~ * ~ |	--> solver loop and post-processing functions
################  ~ * ~ * ~ * ~ |

##########################################################################################
##########################################################################################
def problem_solve(applied_disp,u,du,v):
	# Updated boundary conditions 		
	topBC = DirichletBC(W.sub(1), Constant((-1.0*applied_disp)), top)
		
	bcs = [btmBC, topBC, lftBC, rgtBC] 
	
	# Kinematics
	d = len(u)
	I = Identity(d)             # Identity tensor
	F = I + grad(u)             # Deformation gradient
	F = variable(F)

	psi = 1/2*mu*( inner(F,F) - 3 - 2*ln(det(F)) ) + 1/2*lmbda*(1/2*(det(F)**2 - 1) - ln(det(F)))
	f_int = derivative(psi*dx,u,v)
	f_ext = derivative( dot(B, u)*dx('everywhere') + dot(T, u)*ds , u, v)
	Fboth = f_int - f_ext 
	# Tangent 
	dF = derivative(Fboth, u, du)
	solve(Fboth == 0, u, bcs, J=dF)

	P = diff(psi,F) 
	S = inv(F)*P  
	sig = F*S*F.T*((1/det(F))*I)  
	#vm = sqrt(sig[0,0]*sig[0,0] - sig[0,0]*sig[1,1] + sig[1,1]*sig[1,1] + 3.0*sig[0,1]*sig[0,1])
	
	return u, du, v, f_int, f_ext, psi 
	
to_print = True

def rxn_forces(list_rxn,W,f_int,f_ext):
	x_dofs = W.sub(0).dofmap().dofs()
	y_dofs = W.sub(1).dofmap().dofs()
	f_ext_known = assemble(f_ext)
	f_ext_unknown = assemble(f_int) - f_ext_known
	dof_coords = W.tabulate_dof_coordinates().reshape((-1, 2))
	y_val_min = np.min(dof_coords[:,1]) + 10E-5; y_val_max = np.max(dof_coords[:,1]) - 10E-5
	x_val_min = np.min(dof_coords[:,0]) + 10E-5; x_val_max = np.max(dof_coords[:,0]) - 10E-5
	# --> y reaction on the top and bottom 
	y_top = []; y_btm = [] 
	for kk in y_dofs:
		if dof_coords[kk,1] > y_val_max:
			y_top.append(kk)
		if dof_coords[kk,1] < y_val_min:
			y_btm.append(kk)
	f_sum_top_y = np.sum(f_ext_unknown[y_top])
	f_sum_btm_y = np.sum(f_ext_unknown[y_btm])	
	# --> x reaction on the left and right
	x_lft = []; x_rgt = [] 
	for kk in x_dofs:
		if dof_coords[kk,0] > x_val_max:
			x_rgt.append(kk)
		if dof_coords[kk,0] < x_val_min:
			x_lft.append(kk)
	f_sum_rgt_x = np.sum(f_ext_unknown[x_rgt])
	f_sum_lft_x = np.sum(f_ext_unknown[x_lft])
	
	if to_print: 
		print("y_top, y_btm rxn force:", f_sum_top_y, f_sum_btm_y)
		print("x_lft, x_rgt rxn force:", f_sum_lft_x, f_sum_rgt_x)
	
	# --> check that forces sum to 0 
	sum_x =  f_sum_lft_x + f_sum_rgt_x
	sum_y =  f_sum_top_y + f_sum_btm_y 
	
	if to_print:
		print('sum forces x:',sum_x )
		print('sum forces y:',sum_y )
		
	list_rxn.append([f_sum_lft_x,f_sum_rgt_x,f_sum_top_y,f_sum_btm_y])
	return list_rxn
	
	
def pix_centers(u):
	disps_all_x = np.zeros((28,28))
	disps_all_y = np.zeros((28,28))
	for kk in range(0,28):
		for jj in range(0,28):
			xx = jj + 0.5 # x is columns
			yy = kk + 0.5 # y is rows 
			disps_all_x[kk,jj] = u(xx,yy)[0]
			disps_all_y[kk,jj] = u(xx,yy)[1]
	
	return disps_all_x, disps_all_y


def strain_energy(list_psi, psi):
	val = assemble(psi*dx)
	list_psi.append(val)
	return list_psi

def strain_energy_subtract_first(list_psi):
	first = list_psi[0]
	for kk in range(0,len(list_psi)):
		list_psi[kk] = list_psi[kk] - first 
	return list_psi

# --> set up the loop 
fname_paraview = File(folder_name + "/paraview.pvd")

list_rxn = []

list_psi = [] 

# --> run the loop
disp_val = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] 

fname = folder_name + '/pixel_disp' 

for dd in range(0,len(disp_val)):
	applied_disp = disp_val[dd]
	u, du, v, f_int, f_ext, psi = problem_solve(applied_disp,u,du,v)
	list_rxn = rxn_forces(list_rxn,W,f_int,f_ext)
	#fname_paraview << (u,dd)
	disps_all_x, disps_all_y = pix_centers(u)
	fn_x = fname + '_step' + str(dd) + '_x.txt'
	fn_y = fname + '_step' + str(dd) + '_y.txt'
	np.savetxt(fn_x,disps_all_x)
	np.savetxt(fn_y,disps_all_y)
	list_psi = strain_energy(list_psi, psi)
	
	# --> save paraview file (not necessary most of the time)
	#fname_paraview << u

# --> save reaction forces
fname = folder_name + '/rxn_force.txt'
np.savetxt(fname,np.asarray(list_rxn))

# --> save total (delta) potential energy 
fname = folder_name + '/strain_energy.txt'
list_psi = strain_energy_subtract_first(list_psi)
np.savetxt(fname, np.asarray(list_psi))

##########################################################################################
##########################################################################################










