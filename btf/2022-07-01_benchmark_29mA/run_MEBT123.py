import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(1, os.getcwd())
import btfsim.sim.simulation_main as main
import btfsim.lattice.magUtilities as magutil
import btfsim.bunch.bunchUtilities as butil
from btfsim.lattice.btf_quad_func_factory import BTF_QuadFunctionFactory as quadfunc
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes


start = 'HZ04'
stop = 'VT34a'
mstatefile = 'data/lattice/TransmissionBS34_04212022.mstate'
bunchfilename = 'data/bunch/2Dx3_220701_HZ04_29mA_10M.dat'

irun = 2
runname = Path(__file__).stem
bunchout = '{}_{}_{}_bunch_{}.txt'.format(runname, start, stop, irun)
hist_out_name = 'data/_output/{}_{}_{}_hist_{}.txt'.format(runname, start, stop, irun)


# Parameters
# ------------------------------------------------------------------------------
# Lattice
replace_quads = True
dispersion_flag = False  # if 1, dispersion corrected for in Twiss calculation
nPMQs = 19

# Space charge calcualtion
sclen = 0.01 # [meters]
gridmult= 6
n_bunches = 3 # number of bunches to model
bunch_dec_factor = None  # number of particles reduced by this factor

# Bunch centroid
x0 = y0 = xp0 = yp0 = dE0 = 0.0
# y0 = 6.17  # [mm]
# yp0 = 2.2  # [mrad]


# Set up simulation
# ------------------------------------------------------------------------------
sim = main.simBTF(outdir='data/_output/')
sim.dispersion_flag = int(dispersion_flag)
sim.initLattice(beamline=["MEBT1", "MEBT2", "MEBT3"], mstatename=mstatefile)

# Overlapping nodes: replace quads with analytic model (must come after 
# `lattice.init()` before SC nodes)
z_step = 0.001
quad_names = []  # leaving this empty will change all quads in sequence(s)
for j in range(nPMQs):
    qname = 'MEBT:FQ{:02.0f}'.format(j + 1 + 13)
    quad_names.append(qname)
if replace_quads:
    Replace_Quads_to_OverlappingQuads_Nodes(
        sim.accLattice,
        z_step, 
        accSeq_Names=["MEBT3"], 
        quad_Names=quad_names, 
        EngeFunctionFactory=quadfunc,
    )

sim.initSCnodes(minlen=sclen, solver='fft', gridmult=gridmult, n_bunches=n_bunches)
sim.initBunch(gen="load", file=bunchfilename) 
if bunch_dec_factor is not None and bunch_dec_factor > 1:
    sim.decimateBunch(bunch_dec_factor)
sim.shiftBunch(x0=x0*1e-3, y0=y0*1e-3, xp0=xp0*1e-3, yp0=yp0*1e-3)


# Track bunch
# ------------------------------------------------------------------------------
if type(start) is str:
    startarg = "MEBT:" + start    
else:
    startarg = start
if type(stop) is str:
    stoparg = "MEBT:" + stop
else:
    stoparg = stop
    
sim.run(start=startarg, stop=stoparg, out=bunchout)
sim.tracker.writehist(filename=hist_out_name)