import sys
import re
import os
from qubo_solvers.definitions import DATA_DIR

if len(sys.argv) < 4:
    raise Exception('Need to provide filepath start, end and interior nodes')
    
filepath = sys.argv[1]
start_node, end_node, interior_nodes = sys.argv[2], sys.argv[3], sys.argv[4]
interior_nodes = interior_nodes.split(',')

start_node_without_orientation = start_node[1:]
end_node_without_orientation = end_node[1:]

nodes = [start_node_without_orientation, end_node_without_orientation] + interior_nodes

f = open(filepath, 'r')
filename = os.path.basename(filepath)
dirname = os.path.basename(os.path.dirname(filepath))
write_file = open(f"{DATA_DIR}/{dirname}/{filename}.cropped.{start_node}_{end_node}", 'w')


for line in f:
    match = re.search(r'^H', line)
    if match is not None:
        write_file.write(line)
        continue
    # TODO: match nodes not called "u123"
    match = re.search(r'^S\s(u[0-9]+)\s[ACTG]+\s(.*)\n', line)
    if match is not None:
        node = match.group(1)
        if node in nodes:
            if node == start_node_without_orientation:
                append = "\tST:Z:start"
            elif node == end_node_without_orientation:
                append = "\tST:Z:end"
            else:
                append = ""
            write_file.write(f'S\t{node}\t*\t{match.group(2)}{append}\n')
        continue
    match = re.search(r'^L\s(u[0-9]+)\s[\+\-]\s(u[0-9]+)\s[\+\-]', line)
    if match is not None:
        node_1, node_2 = match.group(1), match.group(2)
        if node_1 in nodes and node_2 in nodes:
            write_file.write(line)
                
f.close()
write_file.close()
        
