import fenics as fe
import numpy as np


class meshrefiner:
    def __init__(self, params, pfproblem, comm):
        self.params= params
        self.dy= params['dy']
        self.max_level= params['max_level']
        self.interface_threshold_gradient= params["interface_threshold_gradient"]
        self.comm = comm
        self.pfproblem= pfproblem 
        self.spacepf= self.pfproblem.spacepf
        self.sv_= self.pfproblem.sv_
        self.mesh_coarse= params['mesh_coarse']

    def value_coor_dof(self):

        (phi_, u_) = fe.split(self.sv_)
        coordinates_of_all = self.spacepf.tabulate_dof_coordinates()
        grad_phi = fe.project(fe.sqrt(fe.dot(fe.grad(phi_), fe.grad(phi_))), self.spacepf)
        phi_value_on_dof = grad_phi.vector().get_local()
        all_Val_dof = self.comm.gather(phi_value_on_dof, root=0)
        all_point = self.comm.gather(coordinates_of_all, root=0)
        # Broadcast the data to all processors
        all_point = self.comm.bcast(all_point, root=0)
        all_Val_dof = self.comm.bcast(all_Val_dof, root=0)
        # Combine the data from all processors
        all_Val_dof_1 = [val for sublist in all_Val_dof for val in sublist]
        all_point_1 = [point for sublist in all_point for point in sublist]
        self.dofcoor = np.array(all_point_1)
        self.valdof = np.array(all_Val_dof_1)

    def coordinates_of_int(self):

        high_gradient_indices = np.where(self.valdof > self.interface_threshold_gradient)[0]
        self.listcoorint = self.dofcoor[high_gradient_indices]

    def mark_mesh(self, coarse_mesh_it):

        mf = fe.MeshFunction("bool", coarse_mesh_it, coarse_mesh_it.topology().dim(), False)
        len_mf = len(mf)
        Cell_Id_List = []
        tree = coarse_mesh_it.bounding_box_tree()
        for Cr in self.listcoorint:
            cell_id = tree.compute_first_entity_collision(fe.Point(Cr))
            if cell_id != 4294967295 and 0 <= cell_id < len_mf:
                Cell_Id_List.append(cell_id)

        Cell_Id_List = np.unique(np.array(Cell_Id_List, dtype=int))
        mf.array()[Cell_Id_List] = True
        return mf
    
    def refine_to_min(self, coarse_mesh_it):

        mf = self.mark_mesh(coarse_mesh_it)
        rfmesh = fe.refine(coarse_mesh_it, mf, redistribute=True)
        return rfmesh

    def refine_mesh(self):
        
        coarse_mesh_it = self.mesh_coarse
        for res in range(self.max_level):
            self.mesh_new = self.refine_to_min(coarse_mesh_it)
            coarse_mesh_it = self.mesh_new

        self.mesh_info = {
            'n_cells': fe.MPI.sum(self.comm, self.mesh_new.num_cells()),
            'hmin': fe.MPI.min(self.comm, self.mesh_new.hmin()),
            'hmax': fe.MPI.max(self.comm, self.mesh_new.hmax()),
            'dx_min': fe.MPI.min(self.comm, self.mesh_new.hmin()) / fe.sqrt(2),
            'dx_max': fe.MPI.max(self.comm, self.mesh_new.hmax()) / fe.sqrt(2),}

    def initialize(self):

        self.value_coor_dof()
        self.coordinates_of_int()
        self.refine_mesh()

                

