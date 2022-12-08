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


# -- 7/1 nominal solution
#mstatefile = '/home/kruisard/Dropbox/Data/mstates/Snapshot_20200701_165155.mstate'
# -- 7/8 model-matched solution
#mstatefile = '/home/kruisard/Dropbox/Data/mstates/200708_ModelMatchedSolution.mstate'
#qdict = {'QH33':6.215,'QV34':-6.140}
# -- 7/13 transmission optimized solution based on model-matched solution
#mstatefile = '/home/kruisard/Dropbox/Data/mstates/Snapshot_20200715_194120.mstate'
# -- 8/31 empirical
mstatefile = 'data/lattice/Snapshot_20200831_155501.mstate'
bunchfilename = 'data/bunch/2Dx3_200902_HZ04_34mA_200k.dat'

irun = 0
runname = Path(__file__).stem
bunchout = '{}_{}_{}_bunch_{}.txt'.format(runname, start, stop, irun)
hist_out_name = 'data/_output/{}_{}_{}_hist_{}.txt'.format(runname, start, stop, irun)


####################################################################################
## Set-up parameters
####################################################################################
replace_quads = 1
dispersionFlag = 0  # if 1, dispersion corrected for in Twiss calculation
nPMQs = 19

sclen = 0.01 # [meters]
gridmult = 6
n_bunches = 3 # number of bunches to model

x0 = 0.0  # [mm]
y0 = 0.0  # [mm]
xp0 = 0. # [mrad]
yp0 = 0. # [mrad] 
dE0 = 0. # Mev


###############################################################################
# -- SETUP SIM
###############################################################################

# -- init sim of BTF (whack into separate sections)
sim = main.simBTF(outdir='data/_output/')

# -- set dispersion flag
sim.dispersionFlag = dispersionFlag

# -- init default lattice w/ mstate solution
#sim.initLattice(beamline=["MEBT1", "MEBT2", "MEBT3"], mstatename=mstatefile)
sim.initLattice(beamline=["MEBT1", "MEBT2", "MEBT3"], mstatename=mstatefile)

# # -- quad adjust
# qdict = {}
# qdict['QV07'] = 5.2#-sim.lat.magdict['QV07']['current']
# qdict['QH08'] = -4.5#-sim.lat.magdict['QH08']['current']
# qdict['QV09'] = 4.5#-sim.lat.magdict['QV09']['current']
# sim.changeQuads(dict=qdict)   
 
    
###############################################################################
# -- OVERLAPPING NODES
###############################################################################
# -- replace quads with analytic model (must come after lattice init but before SC nodes)
quad_names = [] # leaving this empty will change all quads in sequence(s)
for j in range(nPMQs):
    qname = 'MEBT:FQ%02.0f'%(j + 1 + 13)
    quad_names.append(qname)

z_step = 0.001
if replace_quads == 1:
    Replace_Quads_to_OverlappingQuads_Nodes(
        sim.accLattice,
        z_step, 
        accSeq_Names=["MEBT3"], 
        quad_Names=quad_names, 
        EngeFunctionFactory=quadfunc,
    )



# -- set up SC nodes
sim.initSCnodes(minlen=sclen, solver='fft', gridmult=gridmult, n_bunches=n_bunches)

# -- load initial bunch into sims[0]
sim.initBunch(gen="load", file=bunchfilename) 
sim.shiftBunch(x0=x0*1e-3, y0=y0*1e-3, xp0=xp0*1e-3, yp0=yp0*1e-3)

###############################################################################
# -- Run, track bunch
###############################################################################
if type(start) == str:
    startarg = "MEBT:" + start
else:
    startarg = start
if type(stop) == str:
    stoparg = "MEBT:" + stop
else:
    stoparg = stop

# -- run  
sim.run(start=startarg, stop=stoparg, out=bunchout)
sim.tracker.writehist(filename=hist_out_name)