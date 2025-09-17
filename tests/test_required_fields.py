#!/usr/bin/env python3
"""
Test new required fields for dataset.csv compliance.
"""

import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from normalize.standardize import run as standardize_run
from package.exporter import run as export_run

def test_required_fields_end_to_end():
    """Test that all required fields from problem statement are captured and exported."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data with all required fields using SIRENE-like structure
        input_data = {
            # Core identifiers
            'siren': ['123456789', '987654321', '555666777'],
            'siret': ['12345678900001', '98765432100001', '55566677700001'],
            
            # Company information (should map to 'denomination')
            'denominationUniteLegale': ['Expert Comptable Martin SA', 'Cabinet Conseil SARL', 'Avocat & Associés'],
            'activitePrincipaleEtablissement': ['6920Z', '6920Z', '6910Z'],
            'etatAdministratifEtablissement': ['A', 'A', 'A'],
            
            # Legal form and registration date
            'categorieJuridiqueUniteLegale': ['5499', '5498', '5710'],  # SA, SARL, SAS
            'dateCreationUniteLegale': ['2020-01-15', '2019-06-30', '2018-03-20'],
            
            # Address information
            'adresseEtablissement': ['123 Avenue des Champs-Élysées', '456 Rue de la République', '789 Boulevard de la Croisette'],
            'libelleCommuneEtablissement': ['Paris', 'Lyon', 'Cannes'],
            'codePostalEtablissement': ['75008', '69001', '06400'],
            
            # Director information (should map to 'dirigeant_nom' and 'dirigeant_prenom')
            'nomUsageUniteLegale': ['Martin', 'Dupont', 'Durand'],
            'prenomUsuelUniteLegale': ['Jean', 'Marie', 'Pierre'],
            
            # Contact information
            'telephone': ['0142123456', '0472345678', '0493789012'],
            'email': ['contact@expert.fr', 'info@cabinet.com', 'avocat@associes.fr'],
            'website': ['www.expert.fr', 'www.cabinet.com', 'www.avocat.fr'],
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Step 1: Run standardization
        standardize_ctx = {
            'input_path': str(input_path),
            'outdir_path': str(tmpdir),
            'outdir': str(tmpdir),
        }
        
        standardize_result = standardize_run({}, standardize_ctx)
        assert standardize_result['status'] == 'OK', f"Standardization failed: {standardize_result}"
        
        # Check normalized data has all required fields
        normalized_path = tmpdir / "normalized.parquet"
        normalized_df = pd.read_parquet(normalized_path)
        
        # All mandatory fields from problem statement
        required_fields = [
            'siren',               # Company SIREN number ✓
            'denomination',        # Company name (denominationUniteLegale) ✓
            'forme_juridique',     # Legal form (categorieJuridiqueUniteLegale) ✓
            'date_immatriculation', # Registration date (dateCreationUniteLegale) ✓
            'adresse',             # Complete address (adresseEtablissement) ✓
            'code_postal',         # Postal code (codePostalEtablissement) ✓
            'departement',         # Department (extracted from postal code) ✓
            'telephone_norm',      # Phone number (normalized) ✓
            'email',               # Email address ✓
            'website',             # Website ✓
            'dirigeant_nom',       # Director name (nomUsageUniteLegale) ✓
            'dirigeant_prenom'     # Director first name (prenomUsuelUniteLegale) ✓
        ]
        
        missing_fields = [field for field in required_fields if field not in normalized_df.columns]
        assert len(missing_fields) == 0, f"Missing required fields in normalized data: {missing_fields}"
        
        # Verify département extraction
        expected_departments = ['75', '69', '06']  # From postal codes 75008, 69001, 06400
        actual_departments = normalized_df['departement'].tolist()
        assert actual_departments == expected_departments, f"Department extraction failed: expected {expected_departments}, got {actual_departments}"
        
        # Step 2: Run export
        export_ctx = {
            'outdir_path': str(tmpdir),
            'outdir': str(tmpdir),
            'run_id': 'test_required_fields',
            'job_path': None,
        }
        
        export_result = export_run({}, export_ctx)
        assert export_result['status'] == 'OK', f"Export failed: {export_result}"
        
        # Check final dataset.csv
        dataset_path = tmpdir / "dataset.csv"
        assert dataset_path.exists(), "Final dataset.csv not created"
        
        final_df = pd.read_csv(dataset_path)
        
        # All required fields must be in final dataset
        missing_final = [field for field in required_fields if field not in final_df.columns]
        assert len(missing_final) == 0, f"Missing required fields in final dataset.csv: {missing_final}"
        
        # Verify data integrity
        assert len(final_df) == 3, f"Expected 3 records, got {len(final_df)}"
        
        # Verify specific data values
        assert 'Expert Comptable Martin SA' in final_df['denomination'].tolist()
        assert 'Martin' in final_df['dirigeant_nom'].tolist()
        assert 'Jean' in final_df['dirigeant_prenom'].tolist()
        
        # Check phone numbers (they might be stored as integers due to CSV parsing)
        phone_values = final_df['telephone_norm'].astype(str).tolist()
        assert any('+33142123456' in phone or '33142123456' in phone for phone in phone_values), f"Expected phone number not found in {phone_values}"

if __name__ == "__main__":
    test_required_fields_end_to_end()
    print("✅ All required fields test passed!")