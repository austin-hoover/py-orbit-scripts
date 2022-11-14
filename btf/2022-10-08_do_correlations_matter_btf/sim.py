from __future__ import print_function
import sys
import os
import time
import shutil
from pathlib import Path
from pprint import pprint
from pathlib import Path 

import numpy as np
from matplotlib import cm
from psdist import dist

from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes

from btfsim import bunch_utils as bu
from btfsim import lattice_utils as lu
from btfsim import plot
from btfsim import utils
from btfsim.sim import Simulation


# Settings
# ------------------------------------------------------------------------------
start = 0  # start node (name or index)
stop = 'HZ04'  # stop node (name or index)
fio = {'in': {}, 'out': {}}  # to store input/output paths

# Lattice
fio['in']['lattice'] = 'data/lattice/TransmissionBS34_04212022.mstate'

# Bunch
mass = 0.939294  # [GeV/c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV[
freq = 402.5e6  # [Hz]
bunch_current = 0.042  # [A] (if None, do not change)
bunch_dec_factor = 15  # reduce number of macroparticles by this factor
fio['in']['bunch'] = 'data/bunch/realisticLEBT_50mA_42mA_8555k.dat'

# Space charge
sclen = 0.01  # max distance between space charge nodes [m]
gridmult = 6  # grid resolution = 2**gridmult
n_bunches = 3  # number of bunches to model


# Run info
# ------------------------------------------------------------------------------
# Get timestamp and create output directory if it doesn't exixt.
script_name = Path(__file__).stem
datestamp = time.strftime('%Y-%m-%d')
timestamp = time.strftime('%y%m%d%H%M%S')
prefix = '{}-{}-{}-{}'.format(timestamp, script_name, start, stop)
outdir = os.path.join('data/_output/', datestamp)
utils.ensure_path_exists(outdir)

# Save the current git revision hash.
revision_hash = utils.git_revision_hash()
repo_url = utils.git_url()
if revision_hash and repo_url:
    _filename = '{}-{}-git_hash.txt'.format(timestamp, script_name)
    file = open(os.path.join(outdir, _filename), 'w')
    file.write('{}/-/tree/{}'.format(repo_url, revision_hash))
    file.close()
    
# Save time-stamped copy of this file (do not change).
filename = os.path.join(outdir, '{}-{}.py'.format(timestamp, script_name))
shutil.copy(__file__, filename)

# Set output file names.
fio['out']['bunch'] = prefix + '-bunch-{}.dat'.format(stop)
fio['out']['history'] = os.path.join(outdir, prefix + '-history.dat')


# Diagnostics
# ------------------------------------------------------------------------------
def transform(X):
    X = dist.norm_xxp_yyp_zzp(X, scale_emittance=True)
    X = dist.slice_box(X, axis=4, center=0.0, width=0.2)
    return X

plotter = plot.Plotter(
    transform=transform,
    path=outdir,
    default_fig_kws=None, 
    default_save_kws=None,
)

plot_kws = dict(
    bins=70,
    profx=True, 
    profy=True, 
    prof_kws=dict(alpha=0.7, lw=0.7),
    mask_zero=False,
    cmap='viridis',
    # cmap=plot.truncate_cmap(cm.get_cmap('Greys'), 0.12, 1.0),
)

# Set plot limits.
_dims = ['x', 'xp', 'y', 'yp', 'z', 'zp']
_maxs = 6 * [5.0]
# _maxs = [20.0, 40.0, 20.0, 40.0, 20.0, 100.0]
_limits = [(-m, m) for m in _maxs]

# Add plot functions.
for log, axis in zip([True, False], [[(0, 1), (2, 3), (4, 5)], [(0, 2), (4, 0), (4, 2)]]):
    plotter.add_func(
        plot.proj2d_three_column, 
        name = '{}-{}{}_{}{}_{}{}'.format(
            prefix,
            _dims[axis[0][0]], _dims[axis[0][1]],
            _dims[axis[1][0]], _dims[axis[1][1]],
            _dims[axis[2][0]], _dims[axis[2][1]],
        ),
        fig_kws=dict(figsize=(9.0, 3.0), constrained_layout=True), 
        save_kws=dict(), 
        axis=axis, 
        units=False,
        text=True,
        log=log,
        limits=[(_limits[i], _limits[j]) for (i, j) in axis],
        **plot_kws
    )

# Create dict of {parent_node_name: diagnostics_nodes}.
diagnostics_nodes = dict()

        
# Initialize simulation
# ------------------------------------------------------------------------------
sim = Simulation(
    outdir=outdir, 
    # monitor_kws={'plotter': plotter},
)
sim.init_lattice(
    mstatename=os.path.join(os.getcwd(), fio['in']['lattice']),
    beamlines=['MEBT1', 'MEBT2', 'MEBT3'],
)      

# Overlappping PMQ fields
z_step = 0.001
n_pmqs = 19
quad_names = []  # Leaving this empty will replace every quad.
for j in range(n_pmqs):
    quad_names.append('MEBT:FQ{:02.0f}'.format(j + 1 + 13))
Replace_Quads_to_OverlappingQuads_Nodes(
    sim.lattice,
    z_step, 
    accSeq_Names=["MEBT3"], 
    quad_Names=quad_names, 
    EngeFunctionFactory=lu.quad_func_factory,
)

# Add diagnostics nodes.
for parent_node_name, diag_nodes in diagnostics_nodes.items():
    parent_node = sim.lattice.getNodeForName(parent_node_name)
    for diag_node in diag_nodes:
        parent_node.addChildNode(diag_node, node.ENTRANCE)

sim.init_sc_nodes(
    min_dist=sclen, 
    solver='fft', 
    gridmult=gridmult, 
    n_bunches=n_bunches,
    freq=freq,
)

bunch_filename = os.path.join(os.getcwd(), fio['in']['bunch'])
bunch = bu.load(
    os.path.join(os.getcwd(), fio['in']['bunch']),
    file_format='pyorbit',
    verbose=True,
)
bunch.mass(mass)
bunch.charge(charge)
bunch.getSyncParticle().kinEnergy(kin_energy)
if bunch_current:
    bunch = bu.set_current(bunch, current=bunch_current, freq=freq)
if bunch_dec_factor:
    bunch = bu.decimate(bunch, bunch_dec_factor, verbose=True)
sim.init_bunch(bunch)
sim.bunch.dumpBunch(os.path.join(outdir, prefix + '-bunch-init.dat'))


# Run simulation
# ------------------------------------------------------------------------------
# Write node names and positions to file for reference.
file = open(os.path.join(outdir, prefix + '-nodes.txt'), 'w')
file.write('node position')
for node in sim.lattice.getNodes():
    print(node.getName(), node.getPosition())
    file.write('{} {}\n'.format(node.getName(), node.getPosition()))
file.close()

def process_start_stop_arg(arg):
    return "MEBT:{}".format(arg) if type(arg) is str else arg

start = process_start_stop_arg(start)
stop = process_start_stop_arg(stop)
sim.run(start=start, stop=stop, out=fio['out']['bunch'])
sim.monitor.write(filename=fio['out']['history'])