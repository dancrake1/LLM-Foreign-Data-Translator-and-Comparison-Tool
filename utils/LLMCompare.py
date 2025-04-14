import pandas as pd
import numpy as np

def analyze_numeric_changes(curr_file_clean, prev_file_clean, numeric_columns):
    """Analyze changes in numeric columns"""
    print("\nIn analyze_numeric_changes:")
    print("Numeric columns:", numeric_columns.tolist())
    
    if not len(numeric_columns):
        return None
        
    value_changes = pd.DataFrame({
        'Column': numeric_columns,
        'Total_Change': curr_file_clean[numeric_columns].sum() - prev_file_clean[numeric_columns].sum(),
        'Percent_Change': ((curr_file_clean[numeric_columns].sum() - prev_file_clean[numeric_columns].sum()) 
                         / prev_file_clean[numeric_columns].sum().replace(0, np.nan) * 100).round(2)
    })
    
    # Only return if we have meaningful changes
    if value_changes['Total_Change'].any() or value_changes['Percent_Change'].any():
        return value_changes
    return None

def run_compare(output_file_path, prev_file_path, curr_file_path, id_fields, sort_vals):
    """Compare two data files and generate analysis"""
    print("\nStarting comparison...")
    
    # Load files and print initial row counts
    prev_file = pd.read_csv(prev_file_path)
    curr_file = pd.read_csv(curr_file_path)
    
    print("Initial columns - Previous:", prev_file.columns.tolist())
    print("Initial columns - Current:", curr_file.columns.tolist())
    
    # Convert id_fields to list if it's a single string
    id_fields = [id_fields] if isinstance(id_fields, str) else id_fields
    
    print(f"Initial row counts - Previous: {len(prev_file)}, Current: {len(curr_file)}")

    def clean_dataframe(df, file_type):
        # Sort the dataframe
        df = df.sort_values(by=sort_vals, ascending=True)
        
        # Create record identifier from multiple columns if needed
        if len(id_fields) > 1:
            df['record_id'] = df[id_fields].astype(str).apply(lambda x: '-'.join(x), axis=1)
        else:
            df['record_id'] = df[id_fields[0]].astype(str)
        
        # Add counter for records with same identifier
        df['record_num'] = df.groupby('record_id').cumcount() + 1
        
        # Create final identifier that includes both record ID and record number
        df['newIdentifier'] = df['record_id'] + '-' + df['record_num'].astype(str)
        
        # Create sort identifier if different from record identifier
        if sort_vals != id_fields:
            if len(sort_vals) > 1:
                df['sort_id'] = df[sort_vals].astype(str).apply(lambda x: '-'.join(x), axis=1)
            else:
                df['sort_id'] = df[sort_vals[0]].astype(str)
        
        # Clean up intermediate columns
        df.drop(['record_id', 'record_num'] + 
               (['sort_id'] if 'sort_id' in df.columns and sort_vals != id_fields else []), 
               axis=1, inplace=True)
        
        print(f"After cleaning {file_type} file rows: {len(df)}")
        return df
    
    prev_file_clean = clean_dataframe(prev_file, "previous")
    curr_file_clean = clean_dataframe(curr_file, "current")

    # Find common columns and unique columns
    common_columns = prev_file_clean.columns.intersection(curr_file_clean.columns).tolist()
    unique_to_prev = prev_file_clean.columns.difference(curr_file_clean.columns).tolist()
    unique_to_curr = curr_file_clean.columns.difference(prev_file_clean.columns).tolist()
    
    print("\nCommon columns:", common_columns)
    print("Unique to previous:", unique_to_prev)
    print("Unique to current:", unique_to_curr)
    
    # Perform outer merge to ensure we keep all rows
    combined = pd.merge(
        curr_file_clean[common_columns], 
        prev_file_clean[common_columns], 
        how='outer',
        on='newIdentifier',
        suffixes=('_curr', '_prev')
    )
    
    print(f"After merge rows: {len(combined)}")

    # Generate comparison columns
    for col in common_columns:
        if col != 'newIdentifier':
            col_curr = f"{col}_curr"
            col_prev = f"{col}_prev"
            
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(combined[col_curr]) or pd.api.types.is_numeric_dtype(combined[col_prev]):
                # Convert to float for numeric comparison
                combined[col_curr] = pd.to_numeric(combined[col_curr], errors='coerce')
                combined[col_prev] = pd.to_numeric(combined[col_prev], errors='coerce')
                # Consider values within small tolerance as equal
                combined[f'{col}_Match'] = np.isclose(
                    combined[col_curr].fillna(np.nan), 
                    combined[col_prev].fillna(np.nan),
                    equal_nan=True
                )
            else:
                # String comparison for non-numeric columns
                combined[f'{col}_Match'] = (combined[col_curr] == combined[col_prev]) | \
                                         (combined[col_curr].isna() & combined[col_prev].isna())

    # Create record-level summary
    record_summary = pd.DataFrame()
    for field in id_fields:
        curr_records = curr_file[field].nunique()
        prev_records = prev_file[field].nunique()
        record_summary = pd.concat([record_summary, pd.DataFrame({
            'ID Field': [field],
            'Current Unique Values': [curr_records],
            'Previous Unique Values': [prev_records],
            'Difference': [curr_records - prev_records]
        })])

    # Rest of the summary calculations
    match_columns = [f'{col}_Match' for col in common_columns if col != 'newIdentifier']
    curr_columns = [f'{col}_curr' for col in common_columns if col != 'newIdentifier']
    prev_columns = [f'{col}_prev' for col in common_columns if col != 'newIdentifier']

    if 'newIdentifier' in common_columns:
        common_columns.remove('newIdentifier')

    # Create enhanced summary DataFrame with fixed mismatch counting
    counts_df = pd.DataFrame(index=common_columns)
    mismatch_counts = []
    values_only_curr = []
    values_only_prev = []

    for col in common_columns:
        if col != 'newIdentifier':
            col_curr = f"{col}_curr"
            col_prev = f"{col}_prev"
            
            # Count situations where only one file has a value
            curr_only = combined[col_curr].notna() & combined[col_prev].isna()
            prev_only = combined[col_prev].notna() & combined[col_curr].isna()
            
            # Count true mismatches (both values exist and are different)
            both_exist = combined[col_curr].notna() & combined[col_prev].notna()
            mismatches = (~combined[f'{col}_Match']) & both_exist
            
            mismatch_counts.append(mismatches.sum())
            values_only_curr.append(curr_only.sum())
            values_only_prev.append(prev_only.sum())

    # Assign all counts to DataFrame
    counts_df['Different_Values_Between_Files'] = mismatch_counts
    counts_df['NaNCount_curr'] = combined[curr_columns].isna().sum().values
    counts_df['NaNCount_prev'] = combined[prev_columns].isna().sum().values
    counts_df['Values_Only_in_Current'] = values_only_curr
    counts_df['Values_Only_in_Previous'] = values_only_prev

    # Get numeric columns for analysis (common between both files)
    numeric_curr = curr_file_clean.select_dtypes(include=['float', 'int']).columns
    numeric_prev = prev_file_clean.select_dtypes(include=['float', 'int']).columns

    numeric_columns = set(numeric_curr).intersection(set(numeric_prev))  # Convert to set first
    # Remove identifier fields if they happen to be numeric
    for id_field in id_fields:
        numeric_columns.discard(id_field)  # Safe removal
    numeric_columns = pd.Index(numeric_columns)  # Convert back if needed for Pandas operations

    print("\nNumeric columns:", numeric_columns.tolist())

    # Stats summary for numeric columns
    stats_summary = pd.DataFrame({
        'Column': numeric_columns,
        'curr_sum': np.round(curr_file_clean[numeric_columns].sum().tolist(), 2),
        'prev_sum': np.round(prev_file_clean[numeric_columns].sum().tolist(), 2),
        'curr_count': np.round(curr_file_clean[numeric_columns].count().tolist(), 2),
        'prev_count': np.round(prev_file_clean[numeric_columns].count().tolist(), 2),
        'curr_mean': np.round(curr_file_clean[numeric_columns].mean().tolist(), 2),
        'prev_mean': np.round(prev_file_clean[numeric_columns].mean().tolist(), 2),
        'curr_min': np.round(curr_file_clean[numeric_columns].min().tolist(), 2),
        'prev_min': np.round(prev_file_clean[numeric_columns].min().tolist(), 2),
        'curr_max': np.round(curr_file_clean[numeric_columns].max().tolist(), 2),
        'prev_max': np.round(prev_file_clean[numeric_columns].max().tolist(), 2)
    })

    # Create ordered columns list for comparison sheet
    ordered_columns = ['newIdentifier']  # Start with identifier
    for col in common_columns:
        if col != 'newIdentifier':
            ordered_columns.extend([
                f"{col}_curr",
                f"{col}_prev",
                f"{col}_Match"
            ])

    # Create row counts DataFrame
    row_counts = pd.DataFrame({
        'File': ['Previous', 'Current', 'Combined'],
        'Row Count': [len(prev_file), len(curr_file), len(combined)],
    })

    # Create unique columns DataFrame
    unique_cols_df = pd.DataFrame({
        'Unique to Current': unique_to_curr + [None] * (max(len(unique_to_prev), len(unique_to_curr)) - len(unique_to_curr)),
        'Unique to Previous': unique_to_prev + [None] * (max(len(unique_to_prev), len(unique_to_curr)) - len(unique_to_prev))
    })

    # Create a dictionary to store all analysis results
    analysis_results = {
        'standard_sheets': {
            'Full_Compare': combined[ordered_columns],
            'Record_Summary': record_summary,
            'Row_Counts': row_counts,
            'Summary': counts_df.reset_index(),
            'Stats_Summary': stats_summary,
            'Unique_Columns': unique_cols_df
        },
        'custom_analysis': {}
    }

    # Add numeric analysis if available
    print("\nRunning numeric analysis...")
    numeric_analysis = analyze_numeric_changes(curr_file_clean, prev_file_clean, numeric_columns)
    if numeric_analysis is not None:
        analysis_results['custom_analysis']['Numeric_Changes'] = numeric_analysis

    print("\nWriting results to Excel...")
    # Write everything to a single Excel file
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        # Write standard sheets
        for sheet_name, df in analysis_results['standard_sheets'].items():
            print(f"Writing sheet: {sheet_name}")
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Write custom analysis sheets
        print("\nWriting custom analysis sheets...")
        for analysis_type, analysis_data in analysis_results['custom_analysis'].items():
            print(f"Processing: {analysis_type}")
            if isinstance(analysis_data, pd.DataFrame):
                analysis_data.to_excel(writer, sheet_name=analysis_type, index=False)
            elif isinstance(analysis_data, dict):
                # Convert dictionary to DataFrame and write
                pd.DataFrame.from_dict(analysis_data, orient='index').to_excel(
                    writer, sheet_name=analysis_type)

    print(f"\nComparison completed. Row counts - Previous: {len(prev_file)}, "
          f"Current: {len(curr_file)}, Combined: {len(combined)}")
    print("Results saved to", output_file_path)

    return analysis_results