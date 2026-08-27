#!/usr/bin/env python3
from scripts.search import get_default_param_grids
from itertools import product

grids = get_default_param_grids()
model = 'efficientnet_b0'
grid = grids[model]
param_names = list(grid.keys())
param_values = list(grid.values())

print(f'Grid search combinations for {model}:')
print(f'Parameters: {param_names}')
print(f'Values: {param_values}')
total = 1
for vals in param_values:
    total *= len(vals)
print(f'Total combinations: {total}')
print()
print('All combinations:')
for i, combination in enumerate(product(*param_values)):
    params = dict(zip(param_names, combination))
    print(f'{i+1:2d}: {params}')