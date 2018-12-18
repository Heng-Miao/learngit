

import pyopencl as cl
import os
from time import time
import numpy as np











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

        if handcrafted_eos is not None:
            self.eos_table = self.eos.create_table(self.ctx,
                                                self.compile_options)
        elif self.cfg.IEOS == 1:
            self.eos_table = self.eos.create_table(self.ctx,
                                                self.compile_options, nrow=100, ncol=1555)
            chemical_potential_on_hypersf(self.cfg.TFRZ, path,
                                        eos_tyupe='PCE165')
        elif self.cfg.IEOS == 4:
            self.eos_table = self.eos.create_table(self.ctx,
                                                self.compile_options, nrow=4, ncol=1001)
        else:
            self.eos_table = self.eos.create_table(self.ctx,
                                                self.compile_options)
        
        self.efrz = self.eos.f_ed(self.cfg.TFRZ)

        # store 1D and 2D bulk info at each time step
        #
        #
        #
        #
        #
        #
        self.__loadAndBuildCLPrg()

        # define buffer on device side, d_ev1 stores ed, vx, vy,vz
        mf = cl.men_flags
        self.h_ev1 = np.zeros((self.size, 4), self.cfg.real)

        self.d_ev = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1),
                     cl.BUffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)]
        self.d_Src = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)

    def determine_float_size(self, cfg):
        cfg.sz_int = np.dtype('int32').itemsize     # size of (int) in c  [self defination]
    if cfg.use_float32 == True :
        cfg.real = np.float32
        cfg.real4 = array.vec.float4
        cfg.real8 = array.vec.float8
        cfg.sz_real = np.dtype('float32').itemsize   #size of (float) in c
        cfg.sz_real4 = array.vec.float4.itemsize
        cfg.sz_resl8 = array.vec.float8.itemsize
    else :
        cfg.real = np.float64
        cfg.real4 = array.vec.double4
        cfg.real8 = array.vec.double8
        cfg.sz_real = np.dtype('float64').itemsize   #size of  (double) in c
        cfg.sz_real4 = array.vec.double4.itemsize
        cfg.sz_real8 = array.vec.double8.itemsize
    
    def stepUpdate(self, step):
        '''
        Do step update in kernel with KT algorithm
        Args:
            gpu_ev_old: self.d_ev[1] for the 1st step
                        self.d_ev[2] for the 2nd step
            step: the 1st and the 2nd step in Runge-Kutta
        '''
        # update d_Src by KT time splitting, along=1,2,3 for 'x', 'y', 'z'
        # input: gpu_ev_old, tau, size, along_axis
        # output: self.d_Src

        NX = self.cfg.NX
        NY = self.cfg.NY
        NZ = self.cfg.NZ
        BSZ = self.cfg.BSZ

        self.kernel_ideal.kt_src_christoffel(self.queue, (NX*NY*NZ,), None,
                                            self.d_Src, self.d_ev[step], self.eos_table,
                                            self.tau, np.int32(step) ).wait()

        self.kernel_ideal.kt_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                                        self.d_Src, self.d_ev[step], self.eos_table,
                                        self.tau).wait()

        self.kernel_ideal.kt_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                                        self.d_Src, self.d_ev[step], self.eos_table,
                                        self.tau).wait() 

        self.kernel_ideal.kt_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                                        self.d_Src, self.d_ev[step], self.eos_table,
                                        self.tau).wait() 

        # if step=1, T0m' = T0m + d_Src*dt, update d_ev[2]
        # if step=2, T0m = T0m + 0.5*dt*d_Src, update d_ev[1]
        # Notice that d_Src = f(t,x) at step1 and d_Src = (f(t,x)+f(t+dt, x(t+dt))) at step2
        # output: d_ev[] where need_update= 2 for step1 and 1 for step2

        self.kernel_ideal.update_ev(self.queue, (NX*NY*NZ, ), None, 
                                    self.d_ev[3-step], self.d_ev[1], self.d_Src,
                                    self.eos_table, self.tau, np.int32(step) ).wait()

    def max_energy_density(self):
        '''
        Calc the maximum energy density on GPU and output the value
        '''
        self.kernel_reduction.reduction_stage1(self.queue, (256*64,), (256,),
                    self.d_ev[1], self.d_submax, np.int32(self.size) ).wait()
        cl.enqueue_copy(self.queue, self.submax, self.d_submax).wait()
        return self.submax.max()

    def ev_to_host(self):
        '''
        copy energy density and fluid velocity from device to host
        '''
        cl.enqueue_copy(self.queue, self.h_ev1, self.d_ev[1]).wait()

    def update_time(self, loop):
        '''
        update time with TAU0 and loop, convert its type to np.float32 or
        float64 whitch can be used directly as parameter in kernel functions
        '''
        self.tau = self.cfg.real(self.cfg.TAU0 + (loop+1)*self.cfg.DT)

    def  evolve(self, max_loops= ):
        for n in range(max_loops):
            t0 = time()
            self.edmax = self.max_energy_density()
            self.history.append([self.tau, self.edmax ]) 
            print('tau= ', self.tau, 'EdMax= ', self.edmax)
            
            # 1st step in RungeKutta
            self.stepUpdate(step=1)
            # update tau=tau+dtau for the 2nd step in RungeKutta
            self.update_time(loop)
            self.stepUpdate(step=2, jet_eloss_src=jet_eloss_src)
            t1 = time()
            print('one step: {dtime}'.format(dtime = t1-t0))


        self.save(save_hypersf=save_hypersf)                       



def main():
    '''
    set default platform and device in opencl
    '''
    # os.environ['PYOPENCL_CTX'] = '0:0'
    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    from config import cfg, write_config
    print('start ...')
    cfg.IEOS = 1
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 105
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.DZ = 0.16
    cfg.DT = 0.02
    cfg.ntskip = 16
    cfg.nxskip = 2
    cfg.nyskip = 2
    cfg.nzskip = 2
    cfg.Eta_gw = 0.4
    cfg.ImpactParameter = 2.4
    cfg.ETAOS = 0.0
    cfg.TFRZ 0.137

    cfg.Edmax = 55
    cfg.TAU0 =0.4
    cfg.fPathOut = 'results/ideal/'
    cfg.save_to_hdf5 = False
    cfg.BSZ = 64
    write_config(cfg)
    
    ideal = CLIdeal(cfg, gpu_id=1)

    from glauber import glauber
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.compile_options,
                    ideal.d_ev[1])

    t0 = time()
    ideal.evolve(max_loops=2000)
    t1 = time()
    print('finished!Total time for ideal hydro evolution: {dtime'.format(dtime = t1-t0))
    print('Events are written in %s'%cfg.fPathOut)



if __name__ == '__main__':
    main()