#!/bin/bash
set -e

mkdir -p src/data/seed
cd src/data/seed

python3 -c "
from RCAEval.utility import download_re1_dataset
print('Downloading RCAEval RE1 (390MB, ~1 min)...')
download_re1_dataset()
print('Done! Data saved to src/data/seed/')
" 2>/dev/null || python3 -c "
# Fallback when RCAEval package / network unavailable: write a tiny stub manifest
# so downstream generation still works using the bundled hand-crafted templates.
import os, json
os.makedirs('RE1', exist_ok=True)
manifest = {
    'source': 'RCAEval RE1',
    'note': 'Placeholder manifest. Run RCAEval.utility.download_re1_dataset() to fetch the full 390MB dataset.',
    'fault_types': ['CPU', 'MEM', 'DISK', 'DELAY', 'LOSS'],
    'num_cases': 0,
}
with open('RE1/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('Wrote placeholder manifest at src/data/seed/RE1/manifest.json')
print('Scenario templates under src/env/scenarios can still be used without the seed dataset.')
"

cd -
