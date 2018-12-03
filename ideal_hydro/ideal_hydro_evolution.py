

import pyopencl as cl
import os
from time import time











class CLIdeal(object)
    '''The pyopencl version for 3+1D ideal hydro dynamic simulation'''
    def __init__(self, configs, gpu_id=0):
        '''
        param configs: hydrodynamic configurations, from configs import cfg
        param gpu_id: use which gpu for the calculation if there are many per node
        '''
        # create opencl enviroument
        self.cfg = configs
        self.cwd, cwf = os.path.split(__file__)  #split dictory and file
        # create the fPathOut directory if not exists
        path = self.cfg.fPathOut
        if not os.path.exists(path)
            os.makedirs(path)

        # choose proper real, real4, real8 sizes
        self.determine_float_size(self.cfg)
        
        from backend_opencl import OpenCLBackend
        self.backend = OpenCLBackend(self.cfg, gpu_id)

        self.ctx = self.backend.ctx
        self.queue = self.backend.default_queue

        self.size = self.cfg.NX*self.cfg.NY*self.cfg.NZ
        self.tau = self.cfg.real(self.cfg.TAU0)

        self.compile_options = self.__compile_options()

        # set eos, create eos table for interpolation
        # self.eos_table must be before __loadAndBuildCLPrg() to pass
        # table information to definitions
        if handcrafted_eos is None:
            self.eos = Eos(self.cfg.IEOS)
        else:
            self.eos = handcrafted_eos

        # the default muB on hypersf is 0, unless IEOS=1, 'PCE165'
        chemical_potential_on _hypersf(self.cfg.TFRZ, path,
                                        eos_type='ZeroChemcial')



    def determine_float_size(self, cfg):
        cfg.sz_int = np.dtype('int32').itemsize


    def  evolve(self, max_loops= ):
        for n in range(max_loops):
            t0 = time()
            self.edmax = self.max_energy_density()











def main():



if __name__ == '__main__':
    main()