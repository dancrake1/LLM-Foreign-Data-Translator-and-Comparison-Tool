import pandas as pd
from openai_client import OpenAIClient

class ComparisonAnalyzer:
    def __init__(self, llm_params, config):
        self.params = llm_params
        
        # Store OpenAI API key from config
        self.openai_api_key = config.get("openai_api_key")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in config")
            
        # Initialize OpenAI client instead of FlamingoLLMClient
        self.llm_client = OpenAIClient(
            api_key=self.openai_api_key,
            # Optionally specify model variant
            model=OpenAIClient.Model.GPT_4_O  # or other model variants
        )

    def analyze_comparison(self, comparison_data, file1_name, file2_name, mapping1, mapping2):
        filtered_data = {}
        for name, data in comparison_data.items():
            if name != 'Full_Compare':
                if isinstance(data, pd.DataFrame):
                    filtered_data[name] = data.to_dict('records')
                elif isinstance(data, list):
                    filtered_data[name] = data
                else:
                    filtered_data[name] = data

        query = f"""
        You are a data specialist analyzing differences between two datasets. Your task is to provide a comprehensive analysis of how these datasets differ and what implications these differences might have.

        The data includes:
        1. Summary sheets showing key metrics and differences
        2. Detailed comparison of fields across the datasets

        You will analyze:
        
        1. Base Comparison:
           - Summary metrics and statistics between datasets
           - Record-level changes and impacts
           - Data completeness and consistency
        
        2. Value Analysis:
           - Numeric value changes and their significance
           - Differences in key metrics
           - Data patterns that might reveal important insights
        
        3. Field Mappings:
           - Mapping changes and their impact
           - Data type consistency
           - Potential mapping improvements

        Focus on:
        1. Significant Changes:
           - Notable value or count changes (with percentages where meaningful)
           - Distribution shifts that may indicate data quality issues
           - Unexpected changes in typically stable fields
        
        2. Data Quality:
           - Completeness and consistency
           - Anomalies or outliers
           - Mapping-related issues
           - Record-level discrepancies
        
        3. Business Impact:
           - Changes that might affect business decisions
           - Areas requiring attention
           - Recommendations for data quality improvement

        Present your analysis in a clear, structured format for non-technical stakeholders.
        Focus on actionable insights and highlight critical findings without using technical jargon.
        """

        content = {
            "comparison_overview": {
                "file1": file1_name,
                "file2": file2_name
            },
            "comparison_data": filtered_data,
            "mappings": {
                "file1": mapping1,
                "file2": mapping2
            }
        }

        try:
            # Convert content to a string representation that's not too large
            content_str = self._prepare_content_for_api(content)
            
            response = self.llm_client.chat_completions_create(
                messages=[
                    {"role": "system", "content": query},
                    {"role": "user", "content": content_str},
                ],
                stream=False,
                max_tokens=self.params.get("max_new_tokens", 4000),
                temperature=self.params.get("temperature", 0.7),
                top_p=self.params.get("top_p", 1.0)
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error in LLM analysis: {str(e)}")
            
    def _prepare_content_for_api(self, content):
        """
        Prepare content for API by limiting size and formatting properly.
        This helps avoid token limit issues with large datasets.
        """
        # Convert the content to a string representation
        # For large datasets, we may need to summarize or truncate
        
        # Create summary statistics for large data sections
        for key, data in content["comparison_data"].items():
            if isinstance(data, list) and len(data) > 50:
                # Keep only a sample and add summary statistics
                sample = data[:20]  # First 20 items
                content["comparison_data"][key] = {
                    "sample": sample,
                    "total_count": len(data),
                    "summary": f"[Showing 20 out of {len(data)} records]"
                }
        
        # Convert to string representation
        return str(content)

    def close(self):
        """Close the LLM client connection"""
        self.llm_client.close()