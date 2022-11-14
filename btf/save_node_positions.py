"""Write node names and positions to file."""
from __future__ import print_function
import os

from btfsim.sim import Simulation


lattice_filename = 'data/lattice/TransmissionBS34_04212022.mstate'

sim = Simulation()
sim.init_lattice(
    mstatename=os.path.join(os.getcwd(), lattice_filename),
    beamlines=['MEBT1', 'MEBT2', 'MEBT3'],
)           
file = open('data/_output/nodes.txt', 'w')
file.write('node position')
for node in sim.lattice.getNodes():
    print(node.getName(), node.getPosition())
    file.write('{} {}\n'.format(node.getName(), node.getPosition()))
file.close()