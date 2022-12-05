"""Helper functions for PyORBIT scripts."""
from __future__ import print_function
import os
import sys

import numpy as np
from numpy import linalg as la
from scipy import optimize as opt
from tqdm import trange

from bunch import Bunch
from orbit.diagnostics import bunch_coord_array
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import KVDist2D
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.matrix_lattice import MATRIX_Lattice
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot_base import MatrixGenerator
from orbit.twiss.twiss import get_eigtunes
from orbit.twiss.twiss import params_from_transfer_matrix
from orbit.utils.consts import classical_proton_radius
from orbit.twiss.twiss import speed_of_light
from orbit_utils import Matrix


def delete_files_not_folders(path):
    for root, folders, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))
            

def ancestor_folder_path(current_path, ancestor_folder_name):  
    parent_path = os.path.dirname(current_path)
    if parent_path == current_path:
        raise ValueError("Couldn't find ancestor folder.")
    if parent_path.split('/')[-1] == ancestor_folder_name:
        return parent_path
    return ancestor_folder_path(parent_path, ancestor_folder_name)


def tprint(string, indent=4):
    """Print with indent."""    
    print indent * ' ' + str(string)
    
    
def apply(M, X):
    """Apply matrix M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def normalize(X):
    """Normalize all rows of X to unit length."""
    return np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, X)


def symmetrize(M):
    """Return symmetrized version of square upper/lower triangular matrix."""
    return M + M.T - np.diag(M.diagonal())
    
    
def rand_rows(X, n):
    """Return n random elements of X."""
    Xsamp = np.copy(X)
    if n < X.shape[0]:
        idx = np.random.choice(Xsamp.shape[0], n, replace=False)
        Xsamp = Xsamp[idx]
    return Xsamp
    
    
def rotation_matrix(angle):
    """2x2 clockwise rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_matrix_4D(angle):
    """4x4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


# The following three functions are from Tony Yu's blog post: https://tonysyu.github.io/ragged-arrays.html#.YKVwQy9h3OR
def stack_ragged(array_list, axis=0):
    """Stacks list of arrays along first axis.
    
    Example: (25, 4) + (75, 4) -> (100, 4). It also returns the indices at
    which to split the stacked array to regain the original list of arrays.
    """
    lengths = [np.shape(array)[axis] for array in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx
    

def save_stacked_array(filename, array_list, axis=0):
    """Save list of ragged arrays as single stacked array. The index from
    `stack_ragged` is also saved."""
    stacked, idx = stack_ragged(array_list, axis=axis)
    np.savez(filename, stacked_array=stacked, stacked_index=idx)
    
    
def load_stacked_arrays(filename, axis=0):
    """"Load stacked ragged array from .npz file as list of arrays."""
    npz_file = np.load(filename)
    idx = npz_file['stacked_index']
    stacked = npz_file['stacked_array']
    return np.split(stacked, idx, axis=axis)
    
    
def get_perveance(mass, kin_energy, line_density):
    """"Compute dimensionless beam perveance.
    
    Parameters
    ----------
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    line_density : float
        Number density in longitudinal direction [m^-1].
    
    Returns
    -------
    float
        Dimensionless space charge perveance
    """
    gamma = 1 + (kin_energy / mass) # Lorentz factor
    beta = np.sqrt(1 - (1 / gamma)**2) # velocity/speed_of_light
    return (2 * classical_proton_radius * line_density) / (beta**2 * gamma**3)
    
    
def get_intensity(perveance, mass, kin_energy, bunch_length):
    """Return intensity from perveance."""
    gamma = 1 + (kin_energy / mass) # Lorentz factor
    beta = np.sqrt(1 - (1 / gamma)**2) # velocity/speed_of_light
    return beta**2 * gamma**3 * perveance * bunch_length / (2 * classical_proton_radius)
    
    
def get_Brho(mass, kin_energy):
    """Compute magnetic rigidity [T * m]/
    
    Parameters
    ----------
    mass : float
        Particle mass [GeV/c^2].
    kin_energy : float
        Particle kinetic energy [GeV].
    """
    pc = get_pc(mass, kin_energy)
    return 1e9 * (pc / speed_of_light)
    
    
def get_pc(mass, kin_energy):
    """Return momentum * speed_of_light [GeV].
    
    Parameters
    ----------
    mass : float
        Particle mass [GeV/c^2].
    kin_energy : float
        Particle kinetic energy [GeV].
    """
    return np.sqrt(kin_energy * (kin_energy + 2 * mass))


def fodo_lattice(mux, muy, L, fill_fac=0.5, angle=0, start='drift', fringe=False,
                 reverse=False):
    """Create a FODO lattice.
    
    Parameters
    ----------
    mux{y}: float
        The x{y} lattice phase advance [deg]. These are the phase advances
        when the lattice is uncoupled (`angle` == 0).
    L : float
        The length of the lattice.
    fill_fac : float
        The fraction of the lattice occupied by quadrupoles.
    angle : float
        The skew or tilt angle of the quads [deg]. The focusing
        quad is rotated clockwise by angle, and the defocusing quad is
        rotated counterclockwise by angle.
    fringe : bool
        Whether to include nonlinear fringe fields in the lattice.
    start : str
        If 'drift', the lattice will be O-F-O-O-D-O. If 'quad' the lattice will
        be (F/2)-O-O-D-O-O-(F/2).
    reverse : bool
        If True, reverse the lattice elements. This places the defocusing quad
        first.
    
    Returns
    -------
    TEAPOT_Lattice
    """
    angle = np.radians(angle)

    def fodo(k1, k2):
        """Create FODO lattice. k1 and k2 are the focusing strengths of the
        focusing (1st) and defocusing (2nd) quads, respectively.
        """
        # Instantiate elements
        lattice = TEAPOT_Lattice()
        drift1 = teapot.DriftTEAPOT('drift1')
        drift2 = teapot.DriftTEAPOT('drift2')
        drift_half1 = teapot.DriftTEAPOT('drift_half1')
        drift_half2 = teapot.DriftTEAPOT('drift_half2')
        qf = teapot.QuadTEAPOT('qf')
        qd = teapot.QuadTEAPOT('qd')
        qf_half1 = teapot.QuadTEAPOT('qf_half1')
        qf_half2 = teapot.QuadTEAPOT('qf_half2')
        qd_half1 = teapot.QuadTEAPOT('qd_half1')
        qd_half2 = teapot.QuadTEAPOT('qd_half2')
        # Set lengths
        half_nodes = (drift_half1, drift_half2, qf_half1,
                      qf_half2, qd_half1, qd_half2)
        full_nodes = (drift1, drift2, qf, qd)
        for node in half_nodes:
            node.setLength(L * fill_fac / 4)
        for node in full_nodes:
            node.setLength(L * fill_fac / 2)
        # Set quad focusing strengths
        for node in (qf, qf_half1, qf_half2):
            node.addParam('kq', +k1)
        for node in (qd, qd_half1, qd_half2):
            node.addParam('kq', -k2)
        # Create lattice
        if start == 'drift':
            lattice.addNode(drift_half1)
            lattice.addNode(qf)
            lattice.addNode(drift2)
            lattice.addNode(qd)
            lattice.addNode(drift_half2)
        elif start == 'quad':
            lattice.addNode(qf_half1)
            lattice.addNode(drift1)
            lattice.addNode(qd)
            lattice.addNode(drift2)
            lattice.addNode(qf_half2)
        # Other
        if reverse:
            lattice.reverseOrder()
        lattice.set_fringe(fringe)
        lattice.initialize()
        for node in lattice.getNodes():
            name = node.getName()
            if 'qf' in name:
                node.setTiltAngle(+angle)
            elif 'qd' in name:
                node.setTiltAngle(-angle)
        return lattice

    def cost(kvals, correct_tunes, mass=0.93827231, energy=1):
        lattice = fodo(*kvals)
        M = transfer_matrix(lattice, mass, energy)
        return correct_phase_adv - 360. * get_eigtunes(M)

    correct_phase_adv = np.array([mux, muy])
    k0 = np.array([0.5, 0.5]) # ~ 80 deg phase advance
    result = opt.least_squares(cost, k0, args=(correct_phase_adv,))
    k1, k2 = result.x
    return fodo(k1, k2)
    
    
def transfer_matrix(lattice, mass, energy):
    """Shortcut to get transfer matrix from periodic lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].
    
    Returns
    -------
    M : ndarray, shape (4, 4)
        Transverse transfer matrix.
    """
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    one_turn_matrix = matrix_lattice.oneTurnMatrix
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = one_turn_matrix.get(i, j)
    return M
    
    
def twiss_at_entrance(lattice, mass, energy):
    """Get 2D Twiss parameters at lattice entrance.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].
        
    Returns
    -------
    alpha_x, alpha_y, beta_x, beta_y : float
        2D Twiss parameters at lattice entrance.
    """
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    _, arrPosAlphaX, arrPosBetaX = matrix_lattice.getRingTwissDataX()
    _, arrPosAlphaY, arrPosBetaY = matrix_lattice.getRingTwissDataY()
    alpha_x, alpha_y = arrPosAlphaX[0][1], arrPosAlphaY[0][1]
    beta_x, beta_y = arrPosBetaX[0][1], arrPosBetaY[0][1]
    return alpha_x, alpha_y, beta_x, beta_y
    
    
def twiss_throughout(lattice, bunch):
    """Get Twiss parameters throughout lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    bunch : Bunch
        Test bunch to perform tracking.
    
    Returns
    -------
    ndarray
        Columns are: [s, nux, nuy, alpha_x, alpha_x, beta_x, beta_y]
    """
    # Extract Twiss parameters from one turn transfer matrix
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    twiss_x = matrix_lattice.getRingTwissDataX()
    twiss_y = matrix_lattice.getRingTwissDataY()
    # Unpack and convert to ndarrays
    (nux, alpha_x, beta_x), (nuy, alpha_y, beta_y) = twiss_x, twiss_y
    nux, alpha_x, beta_x = np.array(nux), np.array(alpha_x), np.array(beta_x)
    nuy, alpha_y, beta_y = np.array(nuy), np.array(alpha_y), np.array(beta_y)
    # Merge into one array
    s = nux[:, 0]
    nux, alpha_x, beta_x = nux[:, 1], alpha_x[:, 1], beta_x[:, 1]
    nuy, alpha_y, beta_y = nuy[:, 1], alpha_y[:, 1], beta_y[:, 1]
    return np.vstack([s, nux, nuy, alpha_x, alpha_y, beta_x, beta_y]).T
    
    
def get_tunes(lattice, mass, energy):
    """Compute fractional x and y lattice tunes.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].
        
    Returns
    -------
    ndarray, shape (2,)
        Array of [nux, nuy].
    """
    M = transfer_matrix(lattice, mass, energy)
    lattice_params = params_from_transfer_matrix(M)
    nux = lattice_params['frac_tune_x']
    nuy = lattice_params['frac_tune_y']
    return np.array([nux, nuy])

    
def add_node_at_start(lattice, new_node):
    """Add node as child at entrance of first node in lattice."""
    firstnode = lattice.getNodes()[0]
    firstnode.addChildNode(new_node, firstnode.ENTRANCE)


def add_node_at_end(lattice, new_node):
    """Add node as child at end of last node in lattice."""
    lastnode = lattice.getNodes()[-1]
    lastnode.addChildNode(node, lastnode.EXIT)


def add_node_throughout(lattice, new_node, position):
    """Add `new_node` as child of every node in lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        Lattice in which node will be inserted.
    new_node : NodeTEAPOT
        Node to insert.
    position : {'start', 'mid', 'end'}
        Relative location in the lattice nodes to the new node.
        
    Returns
    -------
    None
    """
    loc = {'start': AccNode.ENTRANCE, 
           'mid': AccNode.BODY, 
           'end': AccNode.EXIT}
    
    for node in lattice.getNodes():
        node.addChildNode(new_node, loc[position], 0, AccNode.BEFORE)
        
        
def get_sublattice(lattice, start_node_name=None, stop_node_name=None):
    """Return sublattice from `start_node_name` through `stop_node_name`.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        The original lattice from which to create the sublattice.
    start_node_name, stop_node_name : str
        Names of the nodes in the original lattice to use as the first and
        last node in the sublattice. 
        
    Returns
    -------
    TEAPOT_Lattice
        New lattice consisting of the specified region of the original lattice.
        Note that it is not a copy; changes to the nodes in the new lattice 
        affect the nodes in the original lattice.
    """
    if start_node_name is None:
        start_index = 0
    else:
        start_node = lattice.getNodeForName(start_node_name)
        start_index = lattice.getNodeIndex(start_node)
    if stop_node_name is None:
        stop_index = -1
    else:
        stop_node = lattice.getNodeForName(stop_node_name)
        stop_index = lattice.getNodeIndex(stop_node)
    return lattice.getSubLattice(start_index, stop_index)
        
        
def toggle_spacecharge_nodes(sc_nodes, status='off'):
    """Turn on(off) a set of space charge nodes.
    
    Parameters
    ----------
    sc_nodes : list
        List of space charge nodes. They should be subclasses of
        `SC_Base_AccNode`.
    status : {'on', 'off'}
        Whether to turn the nodes on or off.
    Returns
    -------
    None
    """
    switch = {'on':True, 'off':False}[status]
    for sc_node in sc_nodes:
        sc_node.switcher = switch


def initialize_bunch(mass, energy):
    """Create and initialize Bunch.
    
    Parameters
    ----------
    mass, energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] per bunch particle.
    
    Returns
    -------
    bunch : Bunch
        A Bunch object with the given mass and kinetic energy.
    params_dict : dict
        Dictionary with reference to Bunch.
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(energy)
    params_dict = {'bunch': bunch}
    return bunch, params_dict
        
    
def coasting_beam(kind, n_parts, twiss_params, emittances, length, mass,
                  kin_energy, intensity=0, **kws):
    """Generate bunch with no energy spread and uniform longitudinal density.
    
    Parameters
    ----------
    kind : {'kv', 'gaussian', 'waterbag'}
        The kind of distribution.
    n_parts : int
        Number of macroparticles.
    twiss_params : (ax, ay, bx, by)
        2D Twiss parameters (`ax` means 'alpha x' and so on).
    emittances : (ex, ey)
        Horizontal and vertical r.m.s. emittances.
    length : float
        Bunch length [m].
    mass, kin_energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] per particle.
    intensity : int
        Number of physical particles in the bunch.
    **kws
        Key word arguments for the distribution generator.
    
    Returns
    -------
    bunch : Bunch
        A Bunch object with the specified mass and kinetic energy, filled with
        particles according to the specified distribution.
    params_dict : dict
        Dictionary with reference to Bunch.
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.macroSize(int(intensity / n_parts) if intensity > 0 else 1)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    params_dict = {'bunch': bunch}
    constructors = {'kv':KVDist2D,
                    'gaussian':GaussDist2D,
                    'waterbag':WaterBagDist2D}
    (ax, ay, bx, by), (ex, ey) = twiss_params, emittances
    twissX = TwissContainer(ax, bx, ex)
    twissY = TwissContainer(ay, by, ey)
    dist_generator = constructors[kind](twissX, twissY, **kws)
    for i in range(n_parts):
        x, xp, y, yp = dist_generator.getCoordinates()
        z = np.random.uniform(0, length)
        bunch.addParticle(x, xp, y, yp, z, 0.0)
    return bunch, params_dict
                                                                    
    
def dist_to_bunch(X, bunch, length, deltaE=0.0):
    """Fill bunch with particles.
    
    Parameters
    ----------
    X : ndarray, shape (n_parts, 4)
        Transverse bunch coordinate array.
    bunch : Bunch
        The bunch to populate.
    length : float
        Bunch length. Longitudinal density is uniform.
    deltaE : float
        RMS energy spread in the bunch.
    
    Returns
    -------
    bunch : Bunch
        The modified bunch; don't really need to return this.
    """
    for (x, xp, y, yp) in X:
        z = np.random.uniform(0, length)
        dE = np.random.normal(scale=deltaE)
        bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch
    
    
def track_bunch(bunch, params_dict, lattice, nturns=1, meas_every=0,
                info='coords', progbar=True, mm_mrad=True, transverse_only=False):
    """Track a bunch through the lattice.
    
    Parameters
    ----------
    bunch : Bunch
        The bunch to track.
    params_dict : dict
        Dictionary with reference to `bunch`.
    lattice : TEAPOT_Lattice
        The lattice to track with.
    nturns : int
        Number of times to track track through the lattice.
    meas_every : int
        Store bunch info after every `dump_every` turns. If 0, no info is
        stored.
    info : {'coords', 'cov'}
        If 'coords', the transverse bunch coordinate array is stored. If `cov`,
        the transverse covariance matrix is stored.
    progbar : bool
        Whether to show tqdm progress bar.
    mm_mrad : bool
        Whether to convert from m-rad to mm-mrad.
        
    Returns
    -------
    ndarray, shape (nturns, ?, ?)
        If tracking coords, shape is (nturns, n_parts, 4). If tracking
        covariance matrix, shape is (nturns, 4, 4)
    """
    info_list = []
    turns = trange(nturns) if progbar else range(nturns)
    for turn in turns:
        if meas_every > 0 and turn % meas_every == 0:
            X = bunch_coord_array(bunch, mm_mrad, transverse_only)
            if info == 'coords':
                info_list.append(X)
            elif info == 'cov':
                info_list.append(np.cov(X.T))
        lattice.trackBunch(bunch, params_dict)
    return np.array(info_list)


def split(lattice, max_node_length):
    """Split lattice nodes into parts so no part is longer than max_node_length."""
    for node in lattice.getNodes():
        node_length = node.getLength()
        if node_length > max_node_length:
            node.setnParts(1 + int(node_length / max_node_length))
    return lattice


def set_fringe(lattice, switch):
    """Turn on(off) fringe field calculation for all lattice nodes."""
    for node in lattice.getNodes():
        node.setUsageFringeFieldIN(switch)
        node.setUsageFringeFieldOUT(switch)
    return lattice