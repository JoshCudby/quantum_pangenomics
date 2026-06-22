
### Generate synthetic data or pick real FASTA


  
## repeat 100:  
(More for classical softwares?)

  
### Over many runs, consider rotations of the same input
RotateSequence 
  

### Shotgun Sequence
JamesShotgunSequenceScript

 
### Create node-weighted graph
Syncasm

  

### Solve with each software package some number of times
#### Classical de novo assembly
DeBruijnGraphSolver
SequenceOverlapGraphSolver

  

#### Classical exhaustive
Pathfinder

  

#### Classical QUBO

Mqlib
Gurobi

  

#### Quantum Annealing

Dwave

  

#### QAOA Simulation

QiskitSimulator
PytketSimulator
QOKitSimulator



#### Tensor Network Simulation

CotengraSimulator

  

#### Make FASTAs from paths
OatkPathToFasta

  

#### Score solutions
AssemblathonScoringScripts

  

## endRepeat