import sys
import re

if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = "test.gfa"

nodes=["u5187", "u5188", "u5189", "u5190", "u5191", "u5192", "u5213", "u568", "u569", "u570", "u571", "u572", "u573", "u574", "u575", "u576", "u577", "u578", "u579", "u580", "u581", "u582", "u583", "u584", "u585", "u586", "u587", "u588", "u589", "u590", "u591", "u592", "u593", "u594", "u714", "u715", "u716", "u717"]


f = open(filepath, 'r')
write_file = open(f'{filepath}.cropped.u568_u594', 'w')
for line in f:
    match = re.search(r'^S\s(u[0-9]+)\s', line)
    if match is not None:
        node = match.group(1)
        if node in nodes:
            write_file.write(line)
    else:
        match = re.search(r'^L\s(u[0-9]+)\s[\+\-]\s(u[0-9]+)\s[\+\-]', line)
        if match is not None:
            node_1, node_2 = match.group(1), match.group(2)
            if node_1 in nodes and node_2 in nodes:
                write_file.write(line)
                
f.close()
write_file.close()
        
