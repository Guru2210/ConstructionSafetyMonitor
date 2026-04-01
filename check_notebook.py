import json, ast

with open(r'd:\Construction\train.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
errors = []
for i, c in enumerate(code_cells):
    src = ''.join(c['source'])
    # Strip IPython magic lines (! and %) before syntax check
    clean = '\n'.join(
        line for line in src.splitlines()
        if not line.strip().startswith('!') and not line.strip().startswith('%')
    )
    try:
        ast.parse(clean)
    except SyntaxError as e:
        errors.append(f'Code cell index {i+1}: {e}')

if errors:
    for e in errors:
        print('SYNTAX ERROR -', e)
else:
    print(f'All {len(code_cells)} code cells passed syntax check.')
    print(f'Total cells: {len(nb["cells"])}')
    print('Notebook is ready.')
