from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy


def get_swap_strategy(coupling_map: str, n: int, T: int):
    match coupling_map:
        case 'line':
            extended_swap_strat = ExtendedSwapStrategy.from_line(list(range(n * T)))
        case 'grid':
            extended_swap_strat = ExtendedSwapStrategy.from_grid(n, T)
        case 'all':
            extended_swap_strat = ExtendedSwapStrategy.from_all_to_all(n * T)
        case 'heavy-hex':
            rows, cols = 1, 1
            while 4 * (rows + cols + rows * cols) < n * T:
                if rows < cols:
                    rows += 1
                else:
                    cols += 1
            print(f'Min size to support virtual qubits: {(rows, cols)}')
            extended_swap_strat = ExtendedSwapStrategy.from_heavy_hex(rows, cols)
        case _:
            raise Exception(f'Could not create coupling map: {coupling_map}')
    return extended_swap_strat