"""Track bunch through the BTF lattice."""
from __future__ import print_function
import sys
import os
import time
from pathlib import Path
import numpy as np

from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes

from btfsim.lattice.btf_quad_func_factory import BTF_QuadFunctionFactory as quadfunc
import btfsim.sim.simulation_main as main
import btfsim.bunch.utils as butils


# Setup
# ------------------------------------------------------------------------------
start = 0  # start node (name or index)
stop = 'HZ04'  # stop node (name or index)

# File paths
script_name = Path(__file__).stem
datestamp = time.strftime('%Y-%m-%d')
timestamp = time.strftime('%y%m%d%H%M%S')
outdir = os.path.join('data/_output/', datestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)

fio = dict()
fio['in'] = dict()
fio['out'] = dict()

# Lattice
fio['in']['mstate'] = 'data/lattice/TransmissionBS34_04212022.mstate'
dispersion_flag = False  # dispersion correction in Twiss calculation
replace_quads = True  # use overlapping PMQ field model
nPMQs = 19  # number of permanent-magnet quadrupoles

# Bunch
fio['in']['bunch'] = 'data/bunch/realisticLEBT_50mA_42mA_8555k.dat'
bunch_dec_factor = None
x0 = 0.0  # [m]
y0 = 0.0  # [m]
xp0 = 0.0  # [rad]
yp0 = 0.0  # [rad]
beam_current = 42.0  # current to use in simulation [mA]
beam_current_input = 42.0  # current specified in input bunch file [mA]
decorrelate = True  # presevere only x-x', y-y', z-z' correlations.

# Space charge
use_space_charge = True
sclen = 0.01  # [m]
gridmult = 6
n_bunches = 3  # number of bunches to model


# Initialize simulation
# ------------------------------------------------------------------------------
_base = '{}-{}-{}-{}'.format(timestamp, script_name, start, stop)
fio['out']['bunch'] = _base + '_bunch_{}.dat'.format(stop)
fio['out']['history'] = os.path.join(outdir, _base + '_history.dat')

sim = main.simBTF(outdir=outdir)
sim.dispersion_flag = int(dispersion_flag)
sim.init_lattice(beamline=["MEBT1", "MEBT2", "MEBT3"], mstatename=fio['in']['mstate'])


# Temp: add bunch monitor nodes.
from orbit.teapot import DriftTEAPOT

class BunchMonitorNode(DriftTEAPOT):
    def __init__(self, filename='bunch.dat', name='bunch_monitor'):
        DriftTEAPOT.__init__(self, name)
        self.setLength(0.0)
        self.filename = filename
    def track(self, params_dict):
        bunch = params_dict['bunch']
        bunch.dumpBunch(self.filename)
        
lattice = sim.accLattice
for node in lattice.getNodes():
    if node.getName() in ['MEBT:QH01', 'MEBT:QV02', 'MEBT:QH03', 'MEBT:QV04']:
        filename = os.path.join(outdir, _base + '_bunch_{}.dat'.format(node.getName()))
        bunch_monitor_node = BunchMonitorNode(filename=filename)
        node.addChildNode(bunch_monitor_node, node.ENTRANCE)

for node in lattice.getNodes():
    print(node.getName(), node.getPosition())
    

# Overlapping nodes: replace quads with analytic model. (Must come after 
# `lattice.init()` but before `lattice.init_sc_nodes()`.)
z_step = 0.001
quad_names = []  # Leaving this empty will replace every quad.
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
    
if use_space_charge:
    sim.init_sc_nodes(minlen=sclen, solver='fft', gridmult=gridmult, n_bunches=n_bunches)
    
sim.init_bunch(gen="load", file=os.path.join(os.getcwd(), fio['in']['bunch']))

if decorrelate:
    print('Initial covariance matrix:')
    print(butils.cov(sim.bunch_in))
    sim.decorrelate_bunch()
    print('New covariance matrix:')
    print(butils.cov(sim.bunch_in))

sim.attenuate_bunch(beam_current / beam_current_input)
if bunch_dec_factor is not None and bunch_dec_factor > 1:
    sim.decimate_bunch(bunch_dec_factor)
sim.shift_bunch(x0=x0, y0=y0, xp0=xp0, yp0=yp0)


# Run simulation
# ------------------------------------------------------------------------------
if start == 0:
    sim.dump_bunch(os.path.join(outdir, _base + '_bunch_init.dat'))

def process_start_stop_arg(arg):
    return "MEBT:{}".format(arg) if type(arg) is str else arg

start = process_start_stop_arg(start)
stop = process_start_stop_arg(stop)
sim.run(start=start, stop=stop, out=fio['out']['bunch'])
sim.tracker.write_hist(filename=fio['out']['history'])