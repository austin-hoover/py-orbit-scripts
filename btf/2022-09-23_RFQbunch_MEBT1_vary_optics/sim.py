from __future__ import print_function
import sys
import os
import time
import shutil
from pathlib import Path
from pprint import pprint
from pathlib import Path 

import numpy as np

from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes

from btfsim.lattice import diagnostics
from btfsim.lattice.btf_quad_func_factory import btf_quad_func_factory
from btfsim.sim.sim import Sim
from btfsim.util import utils
import btfsim.bunch.utils as butils


# Setup
# ------------------------------------------------------------------------------
start = 0  # start node (name or index)
stop = 'HZ04'  # stop node (name or index)
switches = {
    'space_charge': True,  # toggle space charge calculation
    'bunch_monitors': False,  # bunch monitor nodes within lattice
    'save_init_bunch': True,   # whether to save initial bunch to file
}

# File paths
fio = {'in': {}, 'out': {}}  # store input/output paths
script_name = Path(__file__).stem
datestamp = time.strftime('%Y-%m-%d')
timestamp = time.strftime('%y%m%d%H%M%S')
outdir = os.path.join('data/_output/', datestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    
# Lattice
fio['in']['mstate'] = 'data/lattice/TransmissionBS34_04212022.mstate'
dispersion_flag = False  # dispersion correction in Twiss calculation
switches['replace_quads'] = True  # use overlapping PMQ field model
n_pmq = 19  # number of permanent-magnet quadrupoles

# Bunch
fio['in']['bunch'] = 'data/bunch/realisticLEBT_50mA_42mA_8555k.dat'
bunch_dec_factor = None
x0 = 0.0  # [m]
y0 = 0.0  # [m]
xp0 = 0.0  # [rad]
yp0 = 0.0  # [rad]
beam_current = 42.0  # current to use in simulation [mA]
beam_current_input = 42.0  # current specified in input bunch file [mA]

# Space charge
sclen = 0.01  # max distance between space charge nodes [m]
gridmult = 6  # grid resolution = 2**gridmult
n_bunches = 3  # number of bunches to model

# Save the current git revision hash.
revision_hash = utils.git_revision_hash()
repo_url = utils.git_url()
if revision_hash and repo_url:
    _filename = '{}-{}-git_hash.txt'.format(timestamp, script_name)
    file = open(os.path.join(outdir, _filename), 'w')
    file.write('{}/-/tree/{}'.format(repo_url, revision_hash))
    file.close()
    
# Save time-stamped copy of this file.
shutil.copy(
    __file__, 
    os.path.join(outdir, '{}-{}.py'.format(timestamp, script_name))
)


# Initialize simulation
# ------------------------------------------------------------------------------
_base = '{}-{}-{}-{}'.format(timestamp, script_name, start, stop)
fio['out']['bunch'] = _base + '-bunch-{}.dat'.format(stop)
fio['out']['history'] = os.path.join(outdir, _base + '-history.dat')

sim = Sim(outdir=outdir)
sim.dispersion_flag = int(dispersion_flag)
sim.init_lattice(
    beamlines=['MEBT1'], 
    mstatename=fio['in']['mstate'],
)

# # Change quad currents.
# print('---------------------------------------')
# print('Changing quad currents!')
# quad_ids = ['QH01', 'QV02', 'QH03', 'QV04']
# for quad_id in quad_ids:
#     current = sim.latgen.magnets[quad_id]['current']
#     spdict = {quad_id: -current}
#     sim.update_quads(units='Amps', **spdict)
# print('---------------------------------------')
    
    
if switches['bunch_monitors']:
    for node in sim.lattice.getNodes():
        if node.getName() in ['MEBT:QH01', 'MEBT:QV02', 'MEBT:QH03', 'MEBT:QV04']:
            filename = os.path.join(outdir, _base + '-bunch-{}.dat'.format(node.getName()))
            bunch_monitor_node = diagnostics.BunchMonitorNode(filename=filename)
            node.addChildNode(bunch_monitor_node, node.ENTRANCE)
for node in sim.lattice.getNodes():
    print(node.getName(), node.getPosition())
    
if switches['space_charge']:
    sim.init_sc_nodes(min_dist=sclen, solver='fft', gridmult=gridmult, n_bunches=n_bunches)
    
sim.init_bunch(
    gen_type='load', 
    bunch_filename=os.path.join(os.getcwd(), fio['in']['bunch']),
    bunch_file_format='pyorbit',
)

## Decorrelate initial beam.
# print('Initial covariance matrix:')
# print(butils.cov(sim.bunch_in))
# sim.decorrelate_bunch()
# print('New covariance matrix:')
# print(butils.cov(sim.bunch_in))

## Transform x-x' <--> y-y'
# for i in range(sim.bunch_in.getSize()):
#     x, xp = sim.bunch_in.x(i), sim.bunch_in.xp(i)
#     y, yp = sim.bunch_in.y(i), sim.bunch_in.yp(i)
#     sim.bunch_in.x(i, y)
#     sim.bunch_in.y(i, x)
#     sim.bunch_in.xp(i, yp)
#     sim.bunch_in.yp(i, xp)

# ## Transform x' --> -x'
# for i in range(sim.bunch_in.getSize()):
#     xp = sim.bunch_in.xp(i)
#     sim.bunch_in.xp(i, -xp)

## Transform y' --> -y'.
for i in range(sim.bunch_in.getSize()):
    yp = sim.bunch_in.yp(i)
    sim.bunch_in.yp(i, -yp)

sim.attenuate_bunch(beam_current / beam_current_input)
if bunch_dec_factor is not None and bunch_dec_factor > 1:
    sim.decimate_bunch(bunch_dec_factor)
sim.shift_bunch(x0=x0, y0=y0, xp0=xp0, yp0=yp0)


# Run simulation
# ------------------------------------------------------------------------------
if switches['save_init_bunch']:
    sim.dump_bunch(os.path.join(outdir, _base + '-bunch-init.dat'))

def process_start_stop_arg(arg):
    return "MEBT:{}".format(arg) if type(arg) is str else arg

start = process_start_stop_arg(start)
stop = process_start_stop_arg(stop)
sim.run(start=start, stop=stop, out=fio['out']['bunch'])
sim.tracker.write_hist(filename=fio['out']['history'])