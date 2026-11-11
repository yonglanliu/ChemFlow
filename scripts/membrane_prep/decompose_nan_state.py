#!/usr/bin/env python
"""Force-group decomposition of an OpenFE nan-error-logs state.

Run in your alchemforge env on any node (CPU is fine):
    python decompose_nan_state.py <dir-with-iteration0-replica0-state0-*.xml>

Prints per-force energies and the atoms carrying the largest forces, so you can see
whether the blow-up is a localized bad interaction (one force group / a few atoms)
or a global dynamics/barostat instability (all forces moderate, box runaway).
"""
import sys
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
sys_xml = next(d.glob("*state0-system.xml"))
sta_xml = next(d.glob("*state0-state.xml"))
print("system:", sys_xml.name, "\nstate: ", sta_xml.name)

system = mm.XmlSerializer.deserialize(sys_xml.read_text())
state = mm.XmlSerializer.deserialize(sta_xml.read_text())

for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)

ctx = mm.Context(system, mm.VerletIntegrator(1.0 * unit.femtosecond),
                 mm.Platform.getPlatformByName("CPU"))
ctx.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
ctx.setPositions(state.getPositions())

print("\n--- per-force potential energy (kJ/mol) ---")
for i, f in enumerate(system.getForces()):
    e = ctx.getState(getEnergy=True, groups={i}).getPotentialEnergy()
    print(f"  [{i}] {type(f).__name__:32s} {e.value_in_unit(unit.kilojoule_per_mole):16.1f}")

st = ctx.getState(getForces=True, getEnergy=True)
print("\ntotal PE:", st.getPotentialEnergy())
F = np.array(st.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
mag = np.linalg.norm(F, axis=1)
order = np.argsort(mag)[::-1][:15]
print("\n--- top-15 atoms by |force| (kJ/mol/nm) ---")
print("atom indices:", order.tolist())
print("magnitudes:  ", [round(float(mag[i]), 1) for i in order])
print("\nNormal is ~1e2-1e3. If a handful are ~1e6+, that spot is the seed.")
print("If all are moderate, the NaN is a dynamics/barostat runaway, not a bad contact.")
