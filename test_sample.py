#!/usr/bin/env python3
"""Create a small test dataset for testing the pipeline."""

import pandas as pd

# Create a small test dataset with accounting firms
test_data = {
    'siren': ['123456789', '987654321'],
    'raison_sociale': ['CABINET EXPERT COMPTABLE MARTIN', 'FIDUCIAIRE COMPTABLE DUBOIS'],
    'naf_code': ['6920Z', '6920Z'],
    'numero_voie': ['10', '25'],
    'type_voie': ['RUE', 'AVENUE'],
    'libelle_voie': ['DU COMMERCE', 'DE LA REPUBLIQUE'],
    'commune': ['PARIS', 'LYON'],
    'code_postal': ['75001', '69001'],
    'department': ['75', '69'],
    'region': ['11', '84'],
    'telephone': ['0142356789', '0478123456'],
    'email': ['', ''],
    'site_web': ['', ''],
    'active': [True, True]
}

df = pd.DataFrame(test_data)
df.to_parquet('/home/runner/work/projects/projects/data/StockEtablissement_utf8.parquet', index=False)
df.to_csv('/home/runner/work/projects/projects/data/StockEtablissement_utf8.csv', index=False)

print(f"Created test dataset with {len(df)} records")
print(df.head())