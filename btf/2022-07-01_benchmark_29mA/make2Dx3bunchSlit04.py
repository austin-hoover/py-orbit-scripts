"""Generate 6D bunch from x-x', y-y', z-z' emittance measurements."""
import sys
import os
import numpy as np

sys.path.insert(1, os.getcwd())
import btfsim.sim.simulation_main as main
import btfsim.bunch.bunchUtilities as butil


# Setup
# ------------------------------------------------------------------------------
mstatefile = 'data/lattice/TransmissionBS34_04212022.mstate'

# 1st emit, 29 mA, TransmissionBS34_04212022.mstate
xfile = 'data/emittance/emittance-data-x-220701191915.csv'
yfile = 'data/emittance/emittance-data-y-220701191040.csv'
# yfile = 'data/emittance/emittance-data-y-merged2.csv'
zfile = 'data/emittance/220726181156-longemittance-emittance-z-dE.csv'  # near x = x' = 0

nparts = 10e6
beam_current = 0.029  # [A]
bunch_frequency= 402.5e6  # [Hz]
ekin = 0.0025 # [GeV]
mass = 0.939294 # [GeV / c^2]
gamma = (mass + ekin) / mass
beta = np.sqrt(gamma**2 - 1.0) / gamma

suffix = '10M'
out_bunch_filename = 'data/bunch/2Dx3_220701_HZ04_29mA_{}.dat'.format(suffix)

threshold = 0.01
method = 'cdf'

# Space charge
sclen = 0.001
gridmult = 6


# Initialize
# ------------------------------------------------------------------------------
sim = main.simBTF()

# Generate 2D bunch from emittance measurements
sim.initBunch(
    gen="2dx3", 
    xfile=xfile,
    yfile=yfile,
    zfile=zfile,
    threshold=threshold,
    nparts=nparts, 
    current=beam_current,
)
sim.bunch_in.dumpBunch(out_bunch_filename)

# Analyze bunch
bcalc = butil.bunchCalculation(sim.bunch_in)
twissx = bcalc.Twiss(plane='x')
print(twissx)
twissy = bcalc.Twiss(plane='y')
print(twissy)
twissz = bcalc.Twiss(plane='z')
print(twissz)

# Propagate via simulation to the RFQ exit.
# sim.initLattice(beamline=["MEBT1", "MEBT2", "MEBT3"], mstatename=mstatefile)
# sim.initApertures(d=0.04)
# sim.initSCnodes(minlen = sclen, solver='fft', gridmult=gridmult)
# out_bunch_name= out_bunch_filename.replace('HZ04', 'RFQ0')
# start = 0.
# stop = "MEBT:HZ04"
# sim.reverse(start=start, stop=stop, out=out_bunch_name)

# # Save output
# hist_out_name = 'data/bunch/btf_hist_toRFQ_{}.txt'.format(suffix)
# sim.tracker.writehist(filename=hist_out_name)
