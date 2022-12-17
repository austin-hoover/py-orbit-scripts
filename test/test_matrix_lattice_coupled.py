from __future__ import print_function
import os
from pprint import pprint 
import sys

from bunch import Bunch 
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.utils.consts import mass_proton


print('Courant-Snyder parameters:')
print('--------------------------')

kin_energy = 1.0  # [GeV]
mass = mass_proton  # [GeV / c^2]

lattice = TEAPOT_Lattice()
lattice_filename = '/home/46h/py-orbit-scripts/data/SNS_ring_nux6.18_nuy6.18.lat'
lattice_seq = 'rnginj'
lattice.readMADX(lattice_filename, lattice_seq)
for node in lattice.getNodes():
    node.setUsageFringeFieldIN(False)
    node.setUsageFringeFieldOUT(False)

test_bunch = Bunch()
test_bunch.mass(mass)
test_bunch.getSyncParticle().kinEnergy(kin_energy)
matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, test_bunch)

params = matrix_lattice.getRingParametersDict()
pprint(params)

print()
print('Lebedev-Bogacz parameters:')
print('--------------------------')

matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(lattice, test_bunch, parameterization='LB')
params = matrix_lattice.getRingParametersDict()
pprint(params)