from plyfile import PlyData, PlyElement
import numpy as np
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC
def wred(str):
    return bcolors.FAIL+str+bcolors.ENDC

def save_colored_pc_ply(coords,colors, path):
    '''

    :param coords:
    :param colors: float within range of 0-1
    :param path:
    :return:
    '''

    # colors *=255
    vertices = np.empty(coords.shape[0],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')
                               ])
    vertices['x'] = coords[:, 0].astype('f4')
    vertices['y'] = coords[:, 1].astype('f4')
    vertices['z'] = coords[:, 2].astype('f4')

    vertices['red'] = colors[:, 0].astype('f4') *255
    vertices['green'] = colors[:, 1].astype('f4') *255
    vertices['blue'] = colors[:, 2].astype('f4') *255

    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)

    ply.write(path)