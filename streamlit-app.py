import streamlit as st
import pandas as pd
import os
import sys
import json
from datetime import datetime, date
import pyperclip  # For clipboard functionality

# Get the directory containing the streamlit app
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, 'utils')
client_run_dir = os.path.join(current_dir, 'clientRun')
data_dir = os.path.join(current_dir, 'Data')  # New Data directory

# Add directories to path
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
if client_run_dir not in sys.path:
    sys.path.append(client_run_dir)

# Import from utils directory
from LLMPydantic import MappingProcessor
from utils.data_comparison import run_data_comparison, check_existing_comparison
from utils.comparison_analyser import ComparisonAnalyzer

# Ensure required directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs("mappedFiles", exist_ok=True)
os.makedirs("metadata", exist_ok=True)  # For storing mapping terms metadata

def get_term_dictionaries():
    """Get list of available term dictionaries"""
    dictionaries = []
    for file in os.listdir("metadata"):
        if file.endswith(".json") and file != "mapping_metadata.json":
            dictionaries.append(file)
    return dictionaries

def load_mapping_terms(dictionary_name="mapping_terms.json"):
    """Load mapping terms from metadata storage"""
    terms_file = os.path.join("metadata", dictionary_name)
    if os.path.exists(terms_file):
        try:
            with open(terms_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Error reading mapping terms file. Creating new terms dictionary.")
    # Default empty structure
    return {"terms": []}

def save_mapping_terms(terms_data, dictionary_name="mapping_terms.json"):
    """Save mapping terms to metadata storage"""
    terms_file = os.path.join("metadata", dictionary_name)
    try:
        with open(terms_file, 'w') as f:
            json.dump(terms_data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving mapping terms: {str(e)}")
        return False

def load_data_files():
    """Load list of files from Data directory"""
    files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                # Get file metadata
                file_size = os.path.getsize(file_path) / (1024 * 1024) # MB
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                files.append({
                    "name": file,
                    "path": file_path,  # Full path
                    "size": f"{file_size:.2f} MB",
                    "modified": file_modified.strftime('%Y-%m-%d %H:%M:%S')
                })
    return sorted(files, key=lambda x: x["name"])

def store_file_metadata(mapping_file, original_path, source_name, category, sheet_name=None, dictionary_name=None):
    """Store metadata about the mapping file and its original source"""
    metadata_file = "mappedFiles/mapping_metadata.json"
    metadata = {}
    
    # Load existing metadata if available
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Error reading metadata file. Creating new metadata.")
    
    # Clean up sheet_name - ensure empty strings are stored as null
    cleaned_sheet_name = sheet_name.strip() if sheet_name else None
    
    # If dictionary_name is not provided, use the current one from session state
    if dictionary_name is None and 'current_dict_name' in st.session_state:
        dictionary_name = st.session_state.current_dict_name
    elif dictionary_name is None:
        dictionary_name = "mapping_terms.json"
    
    # Add or update metadata for this mapping
    metadata[mapping_file] = {
        'original_path': original_path,
        'source_name': source_name,
        'category': category,
        'sheet_name': cleaned_sheet_name,
        'dictionary_name': dictionary_name,  # Store dictionary name in metadata
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save updated metadata
    os.makedirs("mappedFiles", exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


def get_file_metadata(mapping_file):
    """Retrieve metadata for a specific mapping file"""
    metadata_file = "mappedFiles/mapping_metadata.json"
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get(mapping_file, {})
        except json.JSONDecodeError:
            st.error(f"Error reading metadata file.")
    return {}

def clear_mapping_cache():
    """Clear all mapping-related session state"""
    if 'llm_mappings' in st.session_state:
        st.session_state.llm_mappings = []
    if 'completed_mappings' in st.session_state:
        st.session_state.completed_mappings = set()

def remove_mapping(idx):
    """Safely remove a mapping and its associated cache"""
    if 0 <= idx < len(st.session_state.llm_mappings):
        trans = st.session_state.llm_mappings[idx]
        # Remove from completed mappings if present
        if 'File' in trans and trans['File'] in st.session_state.completed_mappings:
            st.session_state.completed_mappings.remove(trans['File'])
        # Remove from queue
        st.session_state.llm_mappings.pop(idx)

def get_config():
    """Get configuration from Streamlit secrets"""
    try:
        return {
            "openai_api_key": st.secrets["llm_credentials"]["openai_api_key"],
        }
    except KeyError as e:
        st.error(f"Missing required configuration in secrets: {str(e)}")
        return None

def load_mapped_files():
    """Load all files from the mappedFiles directory"""
    mapped_files = []
    mapped_dir = "mappedFiles"
    
    if os.path.exists(mapped_dir):
        for file in os.listdir(mapped_dir):
            if file.endswith("_mapping.csv"):
                mapped_files.append(file)
    
    return sorted(mapped_files)

def load_mapping_data(file_path):
    """Load data from a mapping CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

def create_filter_interface(df):
    """Create filter interface for DataFrame columns"""
    filters = {}
    
    # Create filters for each column
    for col in df.columns:
        # Skip columns that are all unique or have too many unique values
        if df[col].nunique() > 50:
            continue
            
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 1:  # Only create filter if there are multiple values
            selected = st.multiselect(
                f"Filter {col}",
                options=unique_vals,
                default=list(unique_vals)
            )
            if selected:
                filters[col] = selected
                
    return filters

def apply_filters(df, filters):
    """Apply selected filters to DataFrame"""
    filtered_df = df.copy()
    for col, values in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(values)]
    return filtered_df

def display_filtered_dataframe(df, title=None):
    """Display DataFrame with filtering options"""
    if title:
        st.write(f"### {title}")
        
    # Initialize session state for filters if not exists
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    
    # Create expander for filters
    with st.expander("Show Filters", expanded=False):
        filters = {}
        # Create filters for each column
        for col in df.columns:
            # Skip columns that are all unique or have too many unique values
            if df[col].nunique() > 50:
                continue
                
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) > 1:  # Only create filter if there are multiple values
                # Use session state to maintain filter selections
                if f"{title}_{col}" not in st.session_state.filters:
                    st.session_state.filters[f"{title}_{col}"] = list(unique_vals)
                
                selected = st.multiselect(
                    f"Filter {col}",
                    options=unique_vals,
                    default=st.session_state.filters[f"{title}_{col}"],
                    key=f"filter_{title}_{col}"  # Unique key for each filter
                )
                
                # Update session state
                st.session_state.filters[f"{title}_{col}"] = selected
                
                if selected:
                    filters[col] = selected
    
    # Apply filters
    filtered_df = df.copy()
    for col, values in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    st.dataframe(filtered_df, use_container_width=True)
    st.write(f"Showing {len(filtered_df)} of {len(df)} rows")

def display_comparison_results(comparison_data, file1_name, file2_name, config=None, llm_params=None):
    """Display comparison results with optional LLM analysis"""
    
    # First display the tabbed data
    tabs = st.tabs(list(comparison_data.keys()))
    for tab, sheet_name in zip(tabs, comparison_data.keys()):
        with tab:
            display_filtered_dataframe(
                comparison_data[sheet_name],
                title=sheet_name
            )
    
    # Add LLM analysis section at the bottom
    if config and llm_params:
        st.divider()
        
        # Initialize session state for analysis if not exists
        if 'comparison_analysis' not in st.session_state:
            st.session_state.comparison_analysis = None
            
        st.write("### AI Analysis of Data Comparison")
        
        # Always show "Generate Analysis" button
        if st.button("Generate Analysis", key="analyze_comparison", use_container_width=True):
            try:
                with st.spinner("Generating analysis..."):
                    # Load the original mapping files
                    df1 = pd.read_csv(os.path.join("mappedFiles", file1_name))
                    df2 = pd.read_csv(os.path.join("mappedFiles", file2_name))
                    
                    # Convert mapping DataFrames to dictionaries
                    mapping1 = df1.to_dict('records')
                    mapping2 = df2.to_dict('records')
                    
                    analyzer = ComparisonAnalyzer(llm_params, config)
                    
                    # Process each summary sheet to maintain all information
                    # Process each sheet except Full_Compare
                    combined_summary = {}
                    for sheet_name, df in comparison_data.items():
                        if sheet_name != 'Full_Compare' and not df.empty:
                            if 'Column' in df.columns:
                                combined_summary[sheet_name] = df.set_index('Column').to_dict('index')
                            else:
                                combined_summary[sheet_name] = df.to_dict('records')
                    
                    analysis = analyzer.analyze_comparison(
                        combined_summary,
                        file1_name,
                        file2_name,
                        mapping1,
                        mapping2
                    )
                    
                    st.session_state.comparison_analysis = analysis
                    analyzer.close()
                    
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
        
        # Display analysis results if available
        if st.session_state.comparison_analysis:
            with st.expander("Analysis Results", expanded=True):
                # Display formatted analysis
                st.markdown(st.session_state.comparison_analysis)
                
                # Add copy functionality with feedback
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("Copy Analysis", key="copy_analysis"):
                        try:
                            pyperclip.copy(st.session_state.comparison_analysis)
                            st.success("Analysis copied to clipboard!")
                        except Exception as e:
                            st.error(f"Error copying to clipboard: {str(e)}")
                    
                    if st.button("Clear Analysis", key="clear_analysis"):
                        st.session_state.comparison_analysis = None

def display_mapping_view():
    if 'comparison_data' not in st.session_state:
        st.session_state.comparison_data = None
    if 'current_files' not in st.session_state:
        st.session_state.current_files = []
    if 'just_saved' not in st.session_state:
        st.session_state.just_saved = None
    if 'last_loaded_dict' not in st.session_state:
        st.session_state.last_loaded_dict = None

    st.title("Mapped Files View")
    mapped_files = load_mapped_files()
    
    if not mapped_files:
        st.warning("No mapped files found in the mappedFiles directory.")
        return
    
    # Handle file selection
    if st.session_state.just_saved:
        current_selection = st.session_state.current_files
        if st.session_state.just_saved not in current_selection:
            current_selection = [st.session_state.just_saved]
        st.session_state.just_saved = None
    else:
        current_selection = st.session_state.current_files if st.session_state.current_files else []

    selected_files = st.multiselect(
        "Select files to view (max 2 files for comparison)",
        mapped_files,
        default=current_selection,
        max_selections=2
    )
    st.session_state.current_files = selected_files
    
    # Track dictionaries loaded from metadata for each file
    dictionaries_from_metadata = []
    
    for idx, file in enumerate(selected_files):
        with st.expander(f"File {idx + 1}: {file}", expanded=True):
            # Get metadata first
            metadata = get_file_metadata(file)
            if metadata:
                # Get the dictionary from metadata or use default
                dictionary_name = metadata.get("dictionary_name", "mapping_terms.json")
                dictionaries_from_metadata.append(dictionary_name)
                
                # If this is the first file or we're switching dictionaries, load it
                if st.session_state.last_loaded_dict != dictionary_name:
                    st.session_state.current_dict_name = dictionary_name
                    st.session_state.current_terms = load_mapping_terms(dictionary_name)
                    st.session_state.last_loaded_dict = dictionary_name
                
                st.write("**File Information:**")
                col1, col2 = st.columns([1, 2])
                col1.markdown("**Original Path:**")
                col2.write(metadata.get("original_path", "N/A"))
                col1.markdown("**Source Name:**")
                col2.write(metadata.get("source_name", metadata.get("portfolio_name", "N/A")))
                col1.markdown("**Category:**")
                col2.write(metadata.get("category", metadata.get("folder", "N/A")))
                col1.markdown("**Sheet Name:**")
                col2.write(metadata.get("sheet_name", "Auto-selected"))
                col1.markdown("**Dictionary:**")  # Add dictionary to displayed metadata
                col2.write(dictionary_name)
                col1.markdown("**Created:**")
                col2.write(metadata.get("creation_date", "N/A"))
                st.divider()
            
            file_path = os.path.join("mappedFiles", file)
            df = load_mapping_data(file_path)
            if df is not None:
                # Load mapping terms from the file's associated dictionary
                mapping_terms_data = load_mapping_terms(st.session_state.current_dict_name)
                mapping_options = [term["term"] for term in mapping_terms_data.get("terms", [])]
                
                # Add "Unknown" option if not already present
                if "Unknown" not in mapping_options:
                    mapping_options.append("Unknown")
                
                # Define valid datatypes from Pydantic model
                datatype_options = ['string', 'integer', 'float', 'boolean', 'date', 'Unknown']
                
                edited_df = st.data_editor(
                    df,
                    column_config={
                        "Mapped": st.column_config.SelectboxColumn(
                            "Mapped",
                            options=mapping_options,
                            required=False
                        ),
                        "Datatype": st.column_config.SelectboxColumn(
                            "Datatype",
                            options=datatype_options,
                            required=True
                        )
                    },
                    hide_index=True,
                    key=f"editor_{idx}",
                    height=600,
                    use_container_width=True
                )
                
                col1, col2, col3 = st.columns([2,1,1])
                with col1:
                    base_name = file.replace('_mapping.csv', '')
                    suffix = st.text_input(f"Add suffix to {base_name}:", key=f"filename_{idx}")
                with col2:
                    if suffix and st.button("Save As", key=f"saveas_{idx}"):
                        # Check if all mapped fields have valid datatypes
                        mapped_rows = edited_df[edited_df['Mapped'].notna()]
                        invalid_types = mapped_rows[mapped_rows['Datatype'].isin(['Unknown', None])]
                        
                        if not invalid_types.empty:
                            # Show which fields need attention
                            invalid_fields = invalid_types['client_cols'].tolist()
                            st.error(f"Please select valid datatypes for the following mapped fields: {', '.join(invalid_fields)}")
                        else:
                            try:
                                new_filename = f"{base_name}_{suffix}_mapping.csv"
                                new_path = os.path.join('mappedFiles', new_filename)
                                
                                # Save the file
                                edited_df.to_csv(new_path, index=False)
                                
                                # Update metadata - preserve the dictionary name from the source file
                                store_file_metadata(
                                    new_filename,
                                    metadata.get("original_path", "N/A"),
                                    metadata.get("source_name", metadata.get("portfolio_name", "N/A")),
                                    metadata.get("category", metadata.get("folder", "N/A")),
                                    metadata.get("sheet_name"),
                                    metadata.get("dictionary_name", st.session_state.current_dict_name)  # Preserve dictionary name
                                )
                                
                                # Set the session state for the next render
                                st.session_state.just_saved = new_filename
                                st.session_state.current_files = selected_files.copy()
                                st.session_state.current_files[idx] = new_filename
                                
                                st.success(f"Saved as: {new_filename}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving file: {str(e)}")
    
    # If we have multiple files with different dictionaries, show a warning
    if len(dictionaries_from_metadata) > 1 and len(set(dictionaries_from_metadata)) > 1:
        st.warning(f"Note: These files use different dictionaries: {', '.join(dictionaries_from_metadata)}. The dictionary from the first file is currently active.")

    # Continue with the comparison code
    if len(selected_files) == 2:
        st.divider()
        st.write("## File Comparison")
        
        try:
            df1 = pd.read_csv(os.path.join("mappedFiles", selected_files[0]))
            df2 = pd.read_csv(os.path.join("mappedFiles", selected_files[1]))
            
            try:
                # Try to get the unique ID field name from metadata
                metadata1 = get_file_metadata(selected_files[0])
                metadata2 = get_file_metadata(selected_files[1])
                
                # Get mapped columns from both dataframes
                mapped_cols1 = df1.loc[df1['Mapped'].notna(), 'Mapped'].unique()
                mapped_cols2 = df2.loc[df2['Mapped'].notna(), 'Mapped'].unique()
                
                # Find common mapped columns
                common_cols = list(set(mapped_cols1).intersection(set(mapped_cols2)))
                
                if not common_cols:
                    st.error("No common mapped fields found between the selected files.")
                    return
                
                if 'last_comparison_files' not in st.session_state or st.session_state.last_comparison_files != tuple(selected_files):
                    st.session_state.comparison_data = None
                    st.session_state.last_comparison_files = tuple(selected_files)
                
                existing_path, existing_data = check_existing_comparison(selected_files[0], selected_files[1])
                
                if st.session_state.comparison_data is None:
                    if existing_data is not None:
                        st.info("Existing comparison found. You can run a new comparison or view the existing one.")
                    
                    # Try to identify ID fields - look for common fields with 'id', 'key', or 'code' in name
                    id_candidates = [col for col in common_cols if any(id_term in col.lower() for id_term in ['id', 'key', 'code', 'number'])]
                    default_id = id_candidates[0] if id_candidates else common_cols[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Change from selectbox to multiselect for ID fields
                        id_fields = st.multiselect(
                            "Select Key Field(s)",
                            options=common_cols,
                            default=[default_id] if default_id in common_cols else [common_cols[0]],
                            help="Select one or more columns that uniquely identify each record"
                        )
                    
                    with col2:
                        # Ensure sort values include all ID fields
                        default_sort = list(set(id_fields))  # Start with ID fields
                        remaining_sort = [x for x in common_cols if x not in default_sort]  # Add other options
                        
                        sort_vals = st.multiselect(
                            "Select Sort Values",
                            options=common_cols,
                            default=default_sort,
                            help="Select columns to sort by. Key fields are included by default."
                        )
                    
                    # Keep the buttons in their own row
                    col1, col2 = st.columns(2)
                    with col1:
                        run_new = st.button("Run New Comparison")
                    with col2:
                        view_existing = existing_data is not None and st.button("View Existing Comparison")
                    
                    if run_new:
                        try:
                            if not id_fields:  # Add validation
                                st.error("Please select at least one key field")
                                return
                                
                            with st.spinner("Running comparison..."):
                                metadata_file = "mappedFiles/mapping_metadata.json"
                                output_path, comparison_data = run_data_comparison(
                                    metadata_file,
                                    selected_files[0],
                                    selected_files[1],
                                    id_fields,  # Now passing the list of ID fields
                                    sort_vals
                                )
                                st.session_state.comparison_data = comparison_data
                                st.success(f"Comparison complete! Results saved to: {output_path}")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error during comparison: {str(e)}")
                            return
                            
                    elif view_existing and existing_data is not None:
                        st.session_state.comparison_data = existing_data
                        st.rerun()
                
                if st.session_state.comparison_data is not None and st.button("Clear Comparison"):
                    st.session_state.comparison_data = None
                    st.session_state.filters = {}
                    st.rerun()
                
                if st.session_state.comparison_data:
                    display_comparison_results(
                        st.session_state.comparison_data,
                        selected_files[0],
                        selected_files[1],
                        config=get_config(),
                        llm_params={
                            "max_new_tokens": 1000,
                            "top_k": 10,
                            "top_p": 0.5,
                            "temperature": 0.01,
                            "repetition_penalty": 1.02,
                            "logprobs": None
                        }
                    )
            except Exception as e:
                st.error(f"Error comparing files: {str(e)}")
                if 'comparison_data' in st.session_state:
                    del st.session_state.comparison_data
                    
        except Exception as e:
            st.error(f"Error loading mapping files: {str(e)}")
            if 'comparison_data' in st.session_state:
                del st.session_state.comparison_data

def display_mapping_terms_editor():
    """Display and manage the mapping terms dictionary"""
    st.subheader("Mapping Terms Dictionary")
    
    # Initialize session state for terms if not exists
    if 'current_terms' not in st.session_state:
        st.session_state.current_terms = {"terms": []}
    
    # Track if we need to add a term (separate from input widget state)
    if 'term_added' not in st.session_state:
        st.session_state.term_added = False
    
    # Track previous dictionary selection
    if 'prev_dict_selection' not in st.session_state:
        st.session_state.prev_dict_selection = "Create New"
    
    # Track current dictionary name
    if 'current_dict_name' not in st.session_state:
        st.session_state.current_dict_name = "mapping_terms.json"
    
    # Select or create a dictionary
    existing_dicts = get_term_dictionaries()
    if existing_dicts:
        dict_options = ["Create New"] + existing_dicts
        selected_dict = st.selectbox(
            "Select Dictionary", 
            dict_options,
            index=0,
            key="dict_selector"
        )
        
        # Check if selection changed
        if selected_dict != st.session_state.prev_dict_selection:
            # If switched to "Create New", clear the terms
            if selected_dict == "Create New":
                st.session_state.current_terms = {"terms": []}
                st.session_state.current_dict_name = "mapping_terms.json"
            else:
                # Automatically load the dictionary when selected
                st.session_state.current_terms = load_mapping_terms(selected_dict)
                st.session_state.current_dict_name = selected_dict
                st.success(f"Loaded {selected_dict} with {len(st.session_state.current_terms.get('terms', []))} terms")
            
            # Update previous selection
            st.session_state.prev_dict_selection = selected_dict
        
        if selected_dict != "Create New":
            # Keep the load button as an option for reloading
            if st.button("Reload Dictionary"):
                st.session_state.current_terms = load_mapping_terms(selected_dict)
                st.session_state.current_dict_name = selected_dict
                st.success(f"Reloaded {selected_dict} with {len(st.session_state.current_terms.get('terms', []))} terms")
    
    # Display current dictionary name
    st.info(f"Active dictionary: **{st.session_state.current_dict_name}**")
    
    # Display current terms
    if st.session_state.current_terms.get("terms"):
        # Convert terms to DataFrame for display
        terms_df = pd.DataFrame(st.session_state.current_terms["terms"])
        st.dataframe(
            terms_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Add a clear button
        if st.button("Clear All Terms"):
            st.session_state.current_terms = {"terms": []}
            st.rerun()
    else:
        st.info("No terms in current dictionary. Use the form below to add terms.")
    
    # Add term form - with unique key based on number of terms
    # This ensures the form "resets" after submission
    term_count = len(st.session_state.current_terms.get("terms", []))
    form_key = f"add_term_form_{term_count}_{selected_dict}"
    
    st.write("### Add New Term")
    
    with st.form(key=form_key):
        col1, col2 = st.columns([2, 4])
        with col1:
            new_term = st.text_input("Term")
        with col2:
            description = st.text_input("Description")
        
        submitted = st.form_submit_button("Add Term")
        
        if submitted and new_term and description:
            # Add the new term to the terms list in session state
            if "terms" not in st.session_state.current_terms:
                st.session_state.current_terms["terms"] = []
            
            # Check if term already exists
            term_exists = any(term["term"] == new_term for term in st.session_state.current_terms["terms"])
            
            if term_exists:
                st.warning(f"Term '{new_term}' already exists in the dictionary.")
            else:
                st.session_state.current_terms["terms"].append({
                    "term": new_term,
                    "description": description
                })
                st.success(f"Added term: {new_term}")
                # Set the term_added flag to trigger a rerun
                st.session_state.term_added = True
                st.rerun()
    
    # Save dictionary section
    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        # If we're creating new, allow editing the name, otherwise use the selected dict name
        dict_name = st.text_input(
            "Dictionary Name",
            value=st.session_state.current_dict_name
        )
    with save_col2:
        if st.button("Save Dictionary", use_container_width=True):
            if st.session_state.current_terms.get("terms"):
                if save_mapping_terms(st.session_state.current_terms, dict_name):
                    st.session_state.current_dict_name = dict_name
                    st.success(f"Saved {len(st.session_state.current_terms['terms'])} terms to {dict_name}")
            else:
                st.error("Cannot save empty dictionary. Add at least one term.")

def display_data_files_list():
    """Display just a list of files from the Data directory with paths"""
    st.subheader("Available Data Files")
    
    # Get file list
    files = load_data_files()
    
    if not files:
        st.info("No files found in the Data directory. Please add files to the 'Data' folder.")
        return
    
    # Create a table showing file paths
    file_paths = []
    for file in files:
        file_name = str(file['name']).startswith('.')
        if not file_name:
            file_paths.append({
                "File Name": file["name"],
                'Size': file['size'],
                "Full Path": file["path"],
                "Last Modified": file["modified"]
            })
    
    # Display the table
    st.dataframe(
        pd.DataFrame(file_paths),
        use_container_width=True,
        hide_index=True
    )
    
    # Direct file selection to populate top form
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_file = st.selectbox(
            "Select file to use", 
            [f["name"] for f in files],
            index=None
        )
    with col2:
        if selected_file and st.button("Use This File", use_container_width=True):
            selected_path = next((f["path"] for f in files if f["name"] == selected_file), None)
            
            # Set session state variables to be used in the main form
            st.session_state["file_path_input"] = selected_path
            st.success(f"Selected: {selected_file}")
            st.info("File path has been populated in the form above")

def data_mapping_view():
    """Main view that combines all functionality - replaces the original page structure"""
    # Initialize session states
    if 'llm_mappings' not in st.session_state:
        st.session_state.llm_mappings = []
    if 'completed_mappings' not in st.session_state:
        st.session_state.completed_mappings = set()
    if 'file_path_input' not in st.session_state:
        st.session_state.file_path_input = ""
    
    # Create sidebar
    st.sidebar.title("Navigation")
    
    # View toggle in sidebar
    view_mode = st.sidebar.radio(
        "Select View",
        ['Data Mapping', 'Mapped Files'],
        key='view_mode'
    )
    
    # Add a clear cache button in the sidebar
    if st.sidebar.button("Clear Mapping Cache"):
        clear_mapping_cache()
        st.rerun()
    
    # Display appropriate view based on selection
    if view_mode == 'Mapped Files':
        display_mapping_view()
    else:
        # Display the data mapping tool
        st.title("Data Mapping Tool")
        
        # LLM Parameters Configuration Section
        with st.expander("AI Parameters Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                max_tokens = st.number_input("Max Tokens", value=5000)
                top_k = st.number_input("Top K", value=10)
                top_p = st.slider("Top P", 0.0, 1.0, 0.5)
            with col2:
                temp = st.slider("Temperature", 0.0, 1.0, 0.01)
                rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.02)
        
        llm_params = {
            "max_new_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temp,
            "repetition_penalty": rep_penalty,
            "logprobs": None
        }
        
        
        # Input form - Uses direct session state value for file path
        with st.form(key='mapping_form'):
            cols = st.columns([3, 2, 2, 1, 1])
            with cols[0]:
                # Use session state directly for the file path
                file_path = st.text_input("File Path", value=st.session_state.file_path_input).strip()
            with cols[1]:
                source_name = st.text_input("Source Name").strip()  # Changed from Portfolio Name
            with cols[2]:
                category = st.text_input("Category").strip()  # Changed from Folder
            with cols[3]:
                # Clean up sheet name input - convert empty string to None
                sheet_name = st.text_input("Sheet (Optional)").strip()
                sheet_name = sheet_name if sheet_name else None  # Convert empty string to None
            with cols[4]:
                submit_button = st.form_submit_button("Add to Queue")
            
            if submit_button:
                if file_path and source_name and category:
                    try:
                        new_mapping = {
                            'Full Path': file_path,
                            'Source Name': source_name,  # Changed from Portfolio Name
                            'Category': category,        # Changed from Folder
                            'Sheet Name': sheet_name,    # Will be None if empty
                            'File': os.path.basename(file_path),
                            'Status': 'Pending'
                        }
                        if new_mapping not in st.session_state.llm_mappings:
                            st.session_state.llm_mappings.append(new_mapping)
                            st.success(f"Added {os.path.basename(file_path)} to mapping queue")
                            
                            # Clear the input values
                            st.session_state.file_path_input = ""
                    except Exception as e:
                        st.error(f"Error adding file to queue: {str(e)}")
                else:
                    st.error("Please fill in all required fields")
        
        # Display added mappings
        if st.session_state.llm_mappings:
            st.write("### Mapping Queue")
            for idx, mapping in enumerate(st.session_state.llm_mappings):
                try:
                    cols = st.columns([3, 2, 2, 1, 1])
                    with cols[0]:
                        st.text(mapping.get('Full Path', 'N/A'))
                    with cols[1]:
                        st.text(mapping.get('Source Name', mapping.get('Portfolio Name', 'N/A')))  # Support legacy data
                    with cols[2]:
                        st.text(mapping.get('Category', mapping.get('Folder', 'N/A')))  # Support legacy data
                    with cols[3]:
                        st.text(mapping.get('Status', 'Pending'))
                    with cols[4]:
                        if st.button("Remove", key=f"remove_{idx}"):
                            remove_mapping(idx)
                            st.rerun()
                except Exception as e:
                    st.error(f"Error displaying mapping {idx}: {str(e)}")
                    remove_mapping(idx)
                    st.rerun()
            
            # Process button
            if st.button("Process All Mappings", use_container_width=True):
                # Get config first
                config = get_config()
                if not config:
                    st.error("Missing required configuration")
                    return
                
                # Get selected term dictionary
                selected_dict = st.session_state.get('current_dict_name', 'mapping_terms.json')
                
                # Show which dictionary is being used
                st.info(f"Using mapping terms from: **{selected_dict}**")
                    
                with st.spinner("Processing mappings..."):
                    # Load mapping terms for LLM
                    mapping_terms = load_mapping_terms(selected_dict)
                    
                    for mapping in st.session_state.llm_mappings:
                        try:
                            if not mapping.get('File'):
                                continue
                                
                            if mapping['File'] not in st.session_state.completed_mappings:
                                # Ensure we use source_name but fall back to portfolio_name for compatibility
                                source_name = mapping.get('Source Name', mapping.get('Portfolio Name', ''))
                                category = mapping.get('Category', mapping.get('Folder', ''))
                                
                                # Ensure source_name is properly escaped for filenames
                                safe_source_name = source_name.replace(' ', '_').replace('/', '_')
                                
                                output_name = f"mappedFiles/{safe_source_name}_{category}_{mapping['File']}_mapping.csv"
                                
                                # Create mappedFiles directory if it doesn't exist
                                os.makedirs("mappedFiles", exist_ok=True)
                                
                                if not os.path.exists(output_name):
                                    processor = None
                                    try:
                                        # Modified to use mapping_terms and updated field names
                                        processor = MappingProcessor(
                                            llm_params,
                                            config,
                                            mapping_terms,  # Pass mapping terms dictionary
                                            mapping['Full Path'],
                                            source_name,     # Changed from portfolio_name
                                            category,        # Changed from folder
                                            sheet_name=mapping.get('Sheet Name')
                                        )
                                        mapping_table = processor.get_final_mapping_table()
                                        mapping_table.to_csv(output_name, index=False)
                                        
                                        # Store metadata after successful mapping - include dictionary name
                                        store_file_metadata(
                                            os.path.basename(output_name),
                                            mapping['Full Path'],
                                            source_name,
                                            category,
                                            mapping.get('Sheet Name'),
                                            selected_dict  # Store the dictionary name in metadata
                                        )
                                        
                                        st.session_state.completed_mappings.add(mapping['File'])
                                        st.success(f"Processed: {mapping['File']}")
                                    finally:
                                        if processor:
                                            processor.close()
                                else:
                                    st.info(f"Skipped {mapping['File']}: Mapping already exists")
                                    st.session_state.completed_mappings.add(mapping['File'])
                                    
                        except Exception as e:
                            error_msg = f"Error processing {mapping.get('File', 'unknown file')}: {str(e)}"
                            st.error(error_msg)
                            mapping['error'] = error_msg
                    
                    # After processing all files, remove successful ones from queue
                    st.session_state.llm_mappings = [
                        mapping for mapping in st.session_state.llm_mappings 
                        if not mapping.get('File') in st.session_state.completed_mappings 
                        or mapping.get('error')
                    ]
        
        # Divider for the bottom section
        st.divider()
        
        # Create two columns for the bottom section
        left_col, right_col = st.columns(2)
        
        # Left column: Just display Data files with paths
        with left_col:
            display_data_files_list()
        
        # Right column: Mapping Terms editor
        with right_col:
            display_mapping_terms_editor()

def main():
    # Application setup
    st.set_page_config(layout="wide", page_title="Data Mapping Tool")  # Changed from Insurance Mapping Tool
    
    # Check for required secrets at startup
    if not get_config():
        st.error("Please configure the required secrets in .streamlit/secrets.toml")
        return
    
    # Start directly with the main view - no setup page
    data_mapping_view()  # Changed from search_view

if __name__ == "__main__":
    main()