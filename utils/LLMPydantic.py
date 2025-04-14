import pandas as pd
from pydantic import BaseModel, create_model
from typing import List, Literal
import json
from data_processing import DataProcessor
from openai_client import OpenAIClient  # Import our new OpenAI client

# Define the Output model
class Output(BaseModel):
    Translated: str
    Mapped: str
    Datatype: Literal['string', 'integer', 'float', 'boolean', 'date', 'Unknown']

class MappingProcessor:
    def __init__(self, llm_params, config, mapping_terms, client_data_path, client_name, year: str, sheet_name=None):
        # Clean up sheet_name input
        self.sheet_name = sheet_name.strip() if sheet_name else None
        
        # Store client info
        self.client_name = client_name
        self.year = year
        self.client_path = client_data_path
        self.params = llm_params
        self.mapping_terms = mapping_terms

        # Store OpenAI API key from config
        self.openai_api_key = config.get("openai_api_key")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in config")

        # Load mapping terms and client data
        self._prepare_mapping_terms()
        self._load_and_clean_client_data(client_data_path)
        print(f'Processing file: {self.client_path}')

        # Initialize OpenAI client
        self.llm_client = OpenAIClient(
            api_key=self.openai_api_key,
            # Optionally specify model variant
            model=OpenAIClient.Model.GPT_4_O  # or other model variants
        )

        # Generate unique value summaries for client data columns
        self.column_summaries = self._extract_unique_values()

        # Create dynamic model, process mappings, and apply to data
        self._create_dynamic_model()
        self._fetch_and_process_mappings()
        self._apply_mappings()

    def _prepare_mapping_terms(self):
        """Prepare mapping terms from the provided dictionary"""
        # If mapping_terms is a dictionary with terms
        if isinstance(self.mapping_terms, dict) and "terms" in self.mapping_terms:
            # Convert terms list to a dictionary format matching the expected structure
            self.standard_mapping_terms = {}
            for term_info in self.mapping_terms["terms"]:
                if "term" in term_info and "description" in term_info:
                    self.standard_mapping_terms[term_info["term"]] = term_info["description"].lower()
        else:
            # Backwards compatibility: If it's a file path, load from Excel
            if isinstance(self.mapping_terms, str) and self.mapping_terms.endswith((".xlsx", ".xls")):
                try:
                    ria = pd.read_excel(self.mapping_terms, sheet_name='LABTerms')
                    self.standard_mapping_terms = dict(zip(ria.iloc[:, 0], ria.iloc[:, 1].str.strip('.').str.lower()))
                    self.standard_mapping_terms = {x: y for x, y in self.standard_mapping_terms.items() if pd.notnull(x)}
                except Exception as e:
                    print(f"Error loading Excel mapping terms: {e}")
                    self.standard_mapping_terms = {}
            else:
                # Default to empty if not valid
                print("Warning: Invalid mapping terms format. Using empty dictionary.")
                self.standard_mapping_terms = {}

    def _load_and_clean_client_data(self, file_path, n_rows=1000, max_rows_to_check=15, columns_to_check=5):
        """Load and clean client data using DataProcessor"""
        data_processor = DataProcessor()
        self.client_data = data_processor.load_and_clean_data(
            file_path, 
            n_rows=n_rows, 
            max_rows_to_check=max_rows_to_check, 
            columns_to_check=columns_to_check,
            sheet_name=self.sheet_name  # Pass sheet_name parameter
        )
        self.client_cols = self.client_data.columns.tolist()
        
        if n_rows is not None:
            self.mapping_df = pd.DataFrame({'client_cols': self.client_cols})
        
        # Extract unique values for LLM processing
        self.column_summaries = self._extract_unique_values()
    
    def _extract_unique_values(self):
        unique_values_dict = {}
        for column in self.client_data.columns:
            try:
                unique_values = self.client_data[column].unique()[:4]  # Get first 4 unique values
                unique_values_dict[column] = {
                    'Data Type': str(self.client_data[column].dtype),
                    'Example Values': list(unique_values)
                }
                if pd.api.types.is_numeric_dtype(self.client_data[column]) or pd.api.types.is_datetime64_any_dtype(self.client_data[column]):
                    unique_values_dict[column].update({
                        'Max': self.client_data[column].max(),
                        'Min': self.client_data[column].min(),
                        'Average': self.client_data[column].mean() if pd.api.types.is_numeric_dtype(self.client_data[column]) else None
                    })
            except Exception as e:
                print(f"Error processing column {column}: {e}")
        return unique_values_dict

    def _create_dynamic_model(self):
        field_definitions = {str(field): (List[Output], None) for field in self.mapping_df['client_cols']}
        self.DynamicModel = create_model('DynamicModel', **field_definitions)

    def _fetch_and_process_mappings(self):
        extra_body = {
            'guided_json': self.DynamicModel.model_json_schema(),
        }

        query = f"""
        You are a data administrator working on foreign data, specializing in language translation. Your task is to map unstructured data fields to a standard set of terms.
        Each client column should map to a JSON array of objects with the structure:
        {{
            "client_column_name": [
                {{"Translated": "translated term here", "Mapped": "mapped common term", "Datatype": "Assumed datatype here"}}
            ]
        }}.

        The common terms with a brief description are as follows: {self.standard_mapping_terms}.
        Your process for mapping should include:
        1. Translating input terms (if necessary) to understand their semantics.
        2. Mapping these terms to the most suitable common term, using descriptions as context.
        3. Determining the datatype based on the meaning and example values provided.

        If no suitable mapped value is found, use "Unknown" for the Mapped field.
        Possible datatypes: string, integer, float, boolean, date. If unsure, use "Unknown" for Datatype.
        
        It is critical that your response be valid JSON that matches the schema provided.
        """
        
        content = f"""
        Client Columns: {self.client_cols}.
        To assist with the meaning of each column, you also have client data summaries, including data types, example values, and metrics on integer values: {self.column_summaries}.

        Once a datamodel has been produced:
        - Reanalyze fields where the "Mapped" output is "Unknown".
        - Use the translated values and common insurance term descriptions to ensure no mappings have been missed.
        """

        # Call the OpenAI API
        self.response = self.llm_client.chat_completions_create(
            messages=[
                {"role": "system", "content": query},
                {"role": "user", "content": content},
            ],
            stream=False, 
            max_tokens=self.params.get("max_new_tokens", 4000),
            temperature=self.params.get("temperature", 0.7),
            top_p=self.params.get("top_p", 1.0),
            extra_body=extra_body
        )

        self.response_output = self.response.choices[0].message.content
        
        # Parse JSON output, handling potential JSON formatting issues
        try:
            self.mapping_dict = json.loads(self.response_output)
        except json.JSONDecodeError:
            # If GPT returns additional text with the JSON, try to extract just the JSON part
            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, self.response_output, re.DOTALL)
            if match:
                self.mapping_dict = json.loads(match.group(1))
            else:
                # Try to find anything that looks like JSON
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, self.response_output, re.DOTALL)
                if match:
                    self.mapping_dict = json.loads(match.group(0))
                else:
                    raise ValueError("Could not parse JSON from OpenAI response")
        
        print('LLM Output Obtained')
        
        # Process mappings into DataFrame
        response_translate = {k: v[0]['Translated'] for k, v in self.mapping_dict.items() if v[0]['Translated'] != 'Unknown'}
        response_mapped = {k: v[0]['Mapped'] for k, v in self.mapping_dict.items() if v[0]['Mapped'] != 'Unknown'}
        response_type = {k: v[0]['Datatype'] for k, v in self.mapping_dict.items() if v[0]['Datatype'] != 'Unknown'}

        # Add mappings to the mapping DataFrame
        self.mapping_df['Translated'] = self.mapping_df['client_cols'].map(response_translate)
        self.mapping_df['Mapped'] = self.mapping_df['client_cols'].map(response_mapped)
        self.mapping_df['Datatype'] = self.mapping_df['client_cols'].map(response_type)

    def _apply_mappings(self):
        # Rename columns in client_data based on the Mapped values
        rename_dict = self.mapping_df.set_index('client_cols')['Mapped'].dropna().to_dict()
        self.client_data.rename(columns=rename_dict, inplace=True)

    def get_final_mapping_table(self):
        return self.mapping_df

    def get_modified_data_subset(self):
        return self.client_data
    
    def print_full_dataset(self):
        # Reload full data if necessary, setting n_rows to None for all rows
        self._load_and_clean_client_data(self.client_path, n_rows=None)
        
        # Apply mappings directly to the loaded dataset
        self._apply_mappings()

        # Get columns that have valid mappings
        matched_cols = self.mapping_df['Mapped'].dropna().unique()

        # Save the processed data to a CSV file
        file_name = self.client_path.split('\\')[-1]
        
        output_path = f"mappedFiles//{self.client_name}_{file_name}_processed.csv"
        self.client_data.to_csv(output_path, columns=matched_cols, index=False)
        print(f"Full dataset saved to {output_path}")

    def get_column_summaries(self):
        return self.column_summaries
    
    def close(self):
        """Closes the LLM client session."""
        self.llm_client.close()
        print('LLM Client closed.')