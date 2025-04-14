import pandas as pd
import os
import json
from LLMCompare import run_compare
from data_processing import load_mapped_data

def generate_comparison_filename(file1_name, file2_name):
    """Generate a consistent filename for comparison results"""
    # Remove _mapping.csv from both filenames
    file1_base = file1_name.replace('_mapping.csv', '')
    file2_base = file2_name.replace('_mapping.csv', '')
    
    return f"mappedFiles/comparisons/{file1_base}_vs_{file2_base}_comparison.xlsx"

def check_existing_comparison(file1_name, file2_name):
    """Check if a comparison file already exists for these two files"""
    comparison_path = generate_comparison_filename(file1_name, file2_name)
    if os.path.exists(comparison_path):
        try:
            # Load all sheets from the Excel file
            comparison_data = pd.read_excel(comparison_path, sheet_name=None)
            return comparison_path, comparison_data
        except Exception:
            return None, None
    return None, None

def run_data_comparison(metadata_file, file1_name, file2_name, pol_code, sort_vals):
    """
    Run comparison between two mapped datasets using metadata from mapping_metadata.json
    Returns both the output path and the comparison data
    """
    try:
        # Read metadata for both files
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        file1_meta = metadata.get(file1_name, {})
        file2_meta = metadata.get(file2_name, {})
        
        if not file1_meta or not file2_meta:
            raise ValueError("Could not find metadata for one or both files")

        # Check if comparison already exists
        existing_path, existing_data = check_existing_comparison(file1_name, file2_name)
        if existing_data is not None:
            return existing_path, existing_data

        # First load and map the data using the DataProcessor
        file1_mapped = load_mapped_data(
            file1_meta['original_path'], 
            f"mappedFiles/{file1_name}",
            apply_datatypes=True  # Enable datatype conversion
        )
        
        file2_mapped = load_mapped_data(
            file2_meta['original_path'], 
            f"mappedFiles/{file2_name}",
            apply_datatypes=True  # Enable datatype conversion
        )
        # Create comparisons directory if it doesn't exist
        os.makedirs("mappedFiles/comparisons", exist_ok=True)
        
        # Generate output filename based on full file names
        output_name = generate_comparison_filename(file1_name, file2_name)
        
        # Save mapped data to temporary CSV files for comparison
        temp_file1 = "mappedFiles/comparisons/temp_file1.csv"
        temp_file2 = "mappedFiles/comparisons/temp_file2.csv"
        
        file1_mapped.to_csv(temp_file1, index=False)
        file2_mapped.to_csv(temp_file2, index=False)
        
        # Run comparison using the mapped files
        run_compare(
            output_file_path=output_name,
            prev_file_path=temp_file1,
            curr_file_path=temp_file2,
            id_fields=pol_code,
            sort_vals=sort_vals
        )
        
        # Clean up temporary files
        os.remove(temp_file1)
        os.remove(temp_file2)
        
        # Load the comparison data we just created
        comparison_data = pd.read_excel(output_name, sheet_name=None)
        
        return output_name, comparison_data
        
    except Exception as e:
        raise Exception(f"Error in comparison: {str(e)}")