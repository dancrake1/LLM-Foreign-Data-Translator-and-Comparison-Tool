import pandas as pd
import chardet
from csv import Sniffer
from collections import Counter
import json
import os

class DataProcessor:
    """
    Handles data loading and cleaning functionality
    """
    def __init__(self):
        self.client_data = None
        self.client_cols = None
        self.column_summaries = None

    def load_and_clean_data(self, file_path, n_rows=None, max_rows_to_check=15, columns_to_check=5, sheet_name=None):
        """
        Load and clean data from CSV or Excel files
        """
        if file_path.endswith('.csv'):
            self._load_csv(file_path, n_rows)
        else:
            self._load_excel(file_path, n_rows, sheet_name)

        self._clean_headers(max_rows_to_check, columns_to_check)
        self._convert_datatypes()
        
        return self.client_data

    def _load_csv(self, file_path, n_rows):
        """Handle CSV file loading with encoding and delimiter detection"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            detected_encoding = chardet.detect(raw_data)['encoding']

        with open(file_path, 'r', encoding=detected_encoding) as f:
            sample = f.read(10000)
            try:
                detected_delimiter = Sniffer().sniff(sample).delimiter
            except:
                common_delimiters = [',', '\t', ';', '|', ' ']
                counts = Counter(sample)
                delimiter_counts = {d: counts[d] for d in common_delimiters}
                detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)

        self.client_data = pd.read_csv(file_path, 
                                     delimiter=detected_delimiter,
                                     encoding=detected_encoding,
                                     nrows=n_rows)

    def _load_excel(self, file_path, n_rows, sheet_name=None):
        """Handle Excel file loading with sheet selection"""
        checking_rows = 100
        try:
            if sheet_name:
                # If sheet name is provided, try to load it directly
                try:
                    self.client_data = pd.read_excel(file_path, sheet_name=sheet_name, nrows=n_rows)
                    return
                except Exception as e:
                    raise ValueError(f"Error reading sheet '{sheet_name}': {str(e)}")
            
            # Original logic for automatic sheet selection
            sheets = pd.read_excel(file_path, sheet_name=None, nrows=checking_rows)
            
            # Filter and sort sheets by data content
            sheet_info = []
            for tab, df in sheets.items():
                if isinstance(df, pd.DataFrame):
                    num_cols = len(df.columns)
                    num_rows = len(df)
                    non_empty_cells = df.notna().sum().sum()
                    sheet_info.append((tab, num_cols, num_rows, non_empty_cells))
            
            sheet_info.sort(key=lambda x: (x[1], x[3]), reverse=True)
            
            if not sheet_info:
                raise ValueError("No valid sheets found in the Excel file")
                
            selected_sheet = sheet_info[0][0]
            self.client_data = pd.read_excel(file_path, sheet_name=selected_sheet, nrows=n_rows)
            
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")

    def _clean_headers(self, max_rows_to_check, columns_to_check):
        """Clean and standardize column headers"""
        unnamed_threshold = int(0.75 * len(self.client_data.columns))
        unnamed_count = sum(1 for col in self.client_data.columns 
                          if isinstance(col, str) and col.startswith("Unnamed") 
                          or pd.isna(col))

        if unnamed_count > unnamed_threshold:
            header_row_index = None
            
            for i in range(min(max_rows_to_check, len(self.client_data))):
                row_values = self.client_data.iloc[i, :columns_to_check].dropna()
                
                if len(row_values) > 0:
                    string_count = sum(1 for x in row_values if isinstance(x, str))
                    if string_count > len(row_values) * 0.75:
                        header_row_index = i
                        break

            if header_row_index is not None:
                column_headers = self.client_data.iloc[header_row_index]
                self.client_data = self.client_data.iloc[header_row_index + 1:].reset_index(drop=True)
                self.client_data.columns = column_headers.values

                # Clean up column names
                self.client_data.columns = [
                    str(col).replace("\n", " ").strip() 
                    if not pd.isna(col) else f"Column_{i}" 
                    for i, col in enumerate(self.client_data.columns)
                ]
            else:
                self.client_data.columns = [f"Column_{i}" for i in range(len(self.client_data.columns))]

        self.client_cols = self.client_data.columns.tolist()

    def _convert_datatypes(self):
        """Convert columns to appropriate datatypes"""
        threshold = 0.8

        def auto_convert_column(column):
            if column.empty:
                return column
                
            numeric_valid = pd.to_numeric(column, errors='coerce').notna().mean()
            if numeric_valid > threshold:
                return pd.to_numeric(column, errors='coerce')

            try:
                datetime_valid = pd.to_datetime(column, errors='coerce', format='%Y-%m-%d').notna().mean()
            except (ValueError, TypeError):
                datetime_valid = pd.to_datetime(column, errors='coerce').notna().mean()
            
            if datetime_valid > threshold:
                return pd.to_datetime(column, errors='coerce')

            return column

        for col in self.client_data.columns:
            try:
                self.client_data[col] = auto_convert_column(self.client_data[col])
            except Exception as e:
                print(f"Warning: Could not convert column {col}: {str(e)}")

    def apply_mappings(self, mapping_df):
        """Apply column mappings from a mapping DataFrame"""
        rename_dict = mapping_df.set_index('client_cols')['Mapped'].dropna().to_dict()
        self.client_data.rename(columns=rename_dict, inplace=True)
        return self.client_data

def load_mapped_data(file_path, mapping_file, apply_datatypes=True):
    """Convenience function to load and map data in one step with datatype conversion"""
    # First read the metadata to get the sheet name
    metadata_file = "mappedFiles/mapping_metadata.json"
    sheet_name = None
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            mapping_basename = os.path.basename(mapping_file)
            if mapping_basename in metadata:
                sheet_name = metadata[mapping_basename].get('sheet_name')
    except Exception as e:
        print(f"Warning: Could not read sheet name from metadata: {str(e)}")

    processor = DataProcessor()
    # Pass the sheet_name to load_and_clean_data
    data = processor.load_and_clean_data(file_path, n_rows=100000, sheet_name=sheet_name)
    mapping_df = pd.read_csv(mapping_file)
    
    # Get mapped columns and their datatypes
    mapping_info = mapping_df[mapping_df['Mapped'].notna()][['client_cols', 'Mapped', 'Datatype']]
    
    # Apply mappings
    rename_dict = mapping_info.set_index('client_cols')['Mapped'].to_dict()
    data.rename(columns=rename_dict, inplace=True)
    
    if apply_datatypes:
        # Create dictionary of column names and their datatypes
        datatype_dict = mapping_info.set_index('Mapped')['Datatype'].to_dict()
        
        # Apply datatypes to each column
        for col, dtype in datatype_dict.items():
            if col in data.columns:
                try:
                    if dtype == 'integer':
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                    elif dtype == 'float':
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    elif dtype == 'boolean':
                        data[col] = data[col].astype(str).str.lower()
                        data[col] = data[col].map({'true': True, '1': True, 'yes': True, 
                                                 'false': False, '0': False, 'no': False})
                    elif dtype == 'date':
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                    # string type doesn't need conversion
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {str(e)}")
    
    # Filter to only include mapped columns
    mapped_cols = mapping_info['Mapped'].unique()
    data = data[mapped_cols]
    
    return data