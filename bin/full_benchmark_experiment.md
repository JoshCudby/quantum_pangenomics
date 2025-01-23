# Generate synthetic data 
JamesSimulateSequenceScript

repeat 100: # More for just classical softwares

# Over many runs, consider rotations of the same input
RotateSequence # Save first 100

# Shotgun Sequence
JamesShotgunSequenceScript

# Create node-weighted graph
JamesGraphCreationScript

# Solve with each software package some number of times

# Classical de novo assembly
DeBruijnGraphSolver
SequenceOverlapGraphSolver

# Classical exhaustive
Pathfinder

# Classical QUBO
Mqlib
Gurobi

# Quantum Annealing
Dwave

# QAOA Simulation
QiskitSimulator
PytketSimulator
QOKitSimulator

# Score solutions
AssemblathonScoringScripts

endRepeat