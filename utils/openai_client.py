from __future__ import annotations

import datetime
from enum import Enum
from openai import OpenAI

class OpenAIClient(OpenAI):
    """
    A client for interacting with OpenAI API, designed as a drop-in replacement
    for FlamingoLLMClient in the existing pipeline.
    """
    
    class Model(Enum):
        GPT_3_5_TURBO = "gpt-3.5-turbo"
        GPT_4 = "gpt-4"
        GPT_4_TURBO = "gpt-4-turbo"
        GPT_4_O = "gpt-4o"  # Latest model as of April 2025
    
    def __init__(self, api_key: str, model: Model = Model.GPT_4_O):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: Your OpenAI API key
            model: The model to use (default: GPT_4_O)
        """
        super().__init__(api_key=api_key)
        self.model_name = model.value
        
    def chat_completions_create(self, messages, stream=False, max_tokens=4000, 
                               temperature=0.7, top_p=1.0, top_k=None, 
                               repetition_penalty=None, logprobs=None, extra_body=None):
        """
        Create a chat completion with OpenAI, maintaining compatibility with FlamingoLLMClient.
        
        Args:
            messages: List of message objects with role and content
            stream: Whether to stream the response
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter (mapped to n if needed)
            repetition_penalty: Repetition penalty (mapped to frequency_penalty)
            logprobs: Whether to return log probabilities
            extra_body: Additional parameters to pass to the API
            
        Returns:
            The OpenAI response object
        """
        # Map any Flamingo-specific parameters to OpenAI parameters
        openai_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        
        # Handle optional parameters
        if repetition_penalty:
            openai_kwargs["frequency_penalty"] = repetition_penalty
        
        if logprobs:
            openai_kwargs["logprobs"] = True
            
        # Handle guided_json if present in extra_body
        if extra_body and 'guided_json' in extra_body:
            schema = extra_body['guided_json']
            # Use OpenAI's response_format parameter
            openai_kwargs["response_format"] = {"type": "json_object"}
            # Add schema instructions to system message
            for i, msg in enumerate(openai_kwargs["messages"]):
                if msg["role"] == "system":
                    openai_kwargs["messages"][i]["content"] += f"\n\nOutput must conform to this JSON schema: {schema}"
                    break
        
        # Call OpenAI API
        return self.chat.completions.create(**openai_kwargs)
        
    def close(self):
        """Close the client session."""
        # No explicit close method needed for the OpenAI client
        pass
