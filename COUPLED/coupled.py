import numpy as np
import fenics as fe
import dolfin as df
from tqdm import tqdm
from classpf import ClassPF
from classrefine import meshrefiner
from classnsipcs import IPCSSolver
comm = df.MPI.comm_world 
rank = df.MPI.rank(comm)
size = df.MPI.size(comm)
fe.set_log_level(fe.LogLevel.ERROR)
file = fe.XDMFFile("GhoshAcoupled.xdmf") 
file.parameters["rewrite_function_mesh"] = True
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
params = {
    'dt': 0.13,#0.13
    "dy": 0.8,#0.8     
    "Nx_aprox": 500,
    "Ny_aprox": 500,
    "max_level": 5,
    "y_solid": 20,
    'w0': 1,
    "Wscale": 1E-8,
    "Tauscale":  2.30808E-8,
    'Tau_0': 1,
    "G": 1E7 , 
    "V": 3E-2 , 
    "ml": 10.5,
    "c0": 5,
    'ep4': 0.03,
    'keq': 0.48,
    'lamda': 1.377,
    'a1': 0.8839,
    'a2': 0.6267,
    'dl': 0.6267* 1.377 ,
    "ds":2.877E-4 ,
    'd0':8E-9,#meter
    "abstol": 1E-6, 
    "reltol": 1E-5, 
    'comm': comm,
    #####################################
    'viscosity_solid': 4.88E-3*1e3,
    'viscosity_liquid':4.88E-3,
    'rho_solid': 8900,
    'rho_liquid': 7810,
    "inflow_profile": fe.Constant((0.02,0)),
    #####################################     
    'nonlinearsolverpf': 'snes',     
    'linearsolverpf': 'mumps',      
    "preconditionerpf": 'hypre_amg',      
    'maximumiterationspf': 50,
    "file": file,
    "Time": 0,
    "T":0,
    "interface_threshold_gradient": 0.0000001,
    "mesh_coarse": None,
}
def refine_mesh(mesh, y_solid, Max_level, dy): 
    mesh_itr = mesh
    for i in range(Max_level):
        mf = fe.MeshFunction("bool", mesh_itr, mesh_itr.topology().dim() , False )
        cells_mesh = fe.cells( mesh_itr )
        index = 0 
        for cell in cells_mesh :

            if  cell.midpoint()[1]    <   y_solid+2*dy : 
                mf.array()[ index ] = True
            index = index + 1 
        mesh_r = fe.refine( mesh_itr, mf )
        # Update for next loop
        mesh_itr = mesh_r
    return mesh_itr 
#################### Define Mesh Domain Parameters ############################
dy_coarse= 2**(params['max_level']) * params['dy']
dy_coarse_init= 2**(4)*params['dy']
nx= (int)(params['Nx_aprox']/dy_coarse)+1
ny= (int)(params["Ny_aprox"]/dy_coarse)+1
nx = nx + 1
ny = ny + 1 
Nx = nx * dy_coarse
Ny = ny * dy_coarse
nx = (int)(Nx / dy_coarse )
ny = (int)(Ny / dy_coarse )
nx_init = (int)(Nx / dy_coarse_init )
ny_init = (int)(Ny / dy_coarse_init )
params['Nx'] = Nx
params["Ny"] = Ny
#############################  END  ################################       
########################## Define Mesh  ##################################
mesh_coarse = fe.RectangleMesh( fe.Point(0, 0), fe.Point(Nx, Ny), nx, ny)
params["mesh_coarse"] = mesh_coarse
coarse_mesh_init = fe.RectangleMesh( fe.Point(0, 0), fe.Point(Nx, Ny), nx_init, ny_init)
mesh = refine_mesh(coarse_mesh_init,params["y_solid"]+20*params["dy"],4, params["dy"])
#############################  END  ####################################
# PF Problem
pfproblem = ClassPF(mesh, params)
pfproblem.initilize()
# NS Problem
nsproblem = IPCSSolver(mesh, params, pfproblem=pfproblem)
nsproblem.initialize()

for it in tqdm(range(int(1e5))):
# for it in range(int(1e5)):

    #.1 PF
    pfproblem.solve()
    #.2 NS
    nsproblem.solve()
    params["Time"]+= params["dt"]
    T = params["Time"]
    if it%1== 0 :
        phi_, c_ = pfproblem.sv_.split(deepcopy=True)
        u_n, p_ = nsproblem.u_n, nsproblem.p_
        phi_.rename("phi", "phi")  
        c_.rename("u", "u")
        u_n.rename("u_n", "u_n")
        p_.rename("p", "p")
        file.write(phi_ , T)  
        file.write(c_ , T)  
        file.write(u_n , T)
        file.write(p_ , T)
        file.close()
    if it% 20== 5: 
        meshreinerobj= meshrefiner(params, pfproblem, comm)
        meshreinerobj.initialize()
        mesh_info = meshreinerobj.mesh_info
        new_mesh = meshreinerobj.mesh_new
        pfold_sv_ = pfproblem.sv_.copy(deepcopy=True)
        pfold_sv = pfproblem.sv.copy(deepcopy=True)
        pfproblem_new = ClassPF(new_mesh, params, nsproblem=nsproblem ,old_sv_=pfold_sv_, old_sv=pfold_sv)
        pfproblem = pfproblem_new
        nsproblem_new = IPCSSolver(new_mesh, params, pfproblem=pfproblem)
        nsproblem = nsproblem_new
        pfproblem.initilize()
        nsproblem.initialize()
        mesh = new_mesh # Update the mesh
    if rank == 0 and it%100 == 0:
        print("Time: ", T)
        print("Iteration: ", it)
    








