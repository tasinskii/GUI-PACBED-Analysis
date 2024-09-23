from abtem import *
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms

#=====PARAMETERS=====
device = 'gpu'
atoms = read("STO-Data/srtio3_100.cif")
cif_height_z = atoms.cell.tolist()[2][2] #angstrom UC z dir distance in .cif
desired_range = (0, 100) # nanometers
detector_max_angle = 30
probe_energy = 200e3
probe_semiangle_cutoff = 27.1
probe_rolloff = 0.05

#edit atoms for desired thickness
height = int(desired_range[1]/cif_height_z)
atoms *= (8, 8, height)

#default params
#gpts = 512, infite projection, .5 slice thickness, kirk param, energy 200e3, semiangle 9.4, rolloff 0.05
potential = Potential(atoms, 
                      gpts=512, 
                      device=device, 
                      projection='infinite', 
                      slice_thickness=cif_height_z, 
                      parametrization='kirkland', 
                      storage=device).build(pbar=True)


detector = PixelatedDetector(max_angle=detector_max_angle)

end = (potential.extent[0] / 8, potential.extent[1] / 8)

scan = GridScan(start=[0, 0], end=end, sampling=0.1)

probe = Probe(energy=probe_energy, semiangle_cutoff=probe_semiangle_cutoff, device=device, rolloff=probe_rolloff)

probe.grid.match(potential)

measurements = [detector.allocate_measurement(probe) for i in range(desired_range[1])]

for indices, positions in scan.generate_positions(max_batch=20, pbar=True):
    probes = probe.build(positions)
    
    for measurement in measurements:
        probes = probes.multislice(potential, pbar=False)
        
        measurement += detector.detect(probes).sum(0)

for i in range(0, desired_range[1]):
    np.save("STO-unaugmented/pacbed-" + str(i) + "nm-SrPbS2", measurements[i].array)



