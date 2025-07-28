"""
Generic Model Integration Service for FM-LLM-Solver

This service provides integration with external LLM APIs like Claude and ChatGPT
for barrier certificate generation as specified in the GuidanceDoc.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests


logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Standard response format for generic models."""
    success: bool
    content: str
    model_name: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GenericModelService:
    """Service for integrating with external LLM APIs."""
    
    def __init__(self, config=None):
        """Initialize the generic model service."""
        self.config = config
        self.api_keys = self._load_api_keys()
        self.model_configs = self._get_model_configurations()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        return {
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY'),
        }
    
    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for each supported model."""
        return {
            'claude-3-sonnet': {
                'provider': 'anthropic',
                'endpoint': 'https://api.anthropic.com/v1/messages',
                'max_tokens': 4096,
                'cost_per_1k_tokens': {'input': 0.003, 'output': 0.015},
                'display_name': 'Claude 3 Sonnet',
                'description': 'Anthropic Claude 3 Sonnet - Excellent for complex reasoning'
            },
            'claude-3-haiku': {
                'provider': 'anthropic',
                'endpoint': 'https://api.anthropic.com/v1/messages',
                'max_tokens': 4096,
                'cost_per_1k_tokens': {'input': 0.00025, 'output': 0.00125},
                'display_name': 'Claude 3 Haiku',
                'description': 'Anthropic Claude 3 Haiku - Fast and cost-effective'
            },
            'gpt-4': {
                'provider': 'openai',
                'endpoint': 'https://api.openai.com/v1/chat/completions',
                'max_tokens': 4096,
                'cost_per_1k_tokens': {'input': 0.03, 'output': 0.06},
                'display_name': 'GPT-4',
                'description': 'OpenAI GPT-4 - Advanced reasoning and analysis'
            },
            'gpt-4-turbo': {
                'provider': 'openai', 
                'endpoint': 'https://api.openai.com/v1/chat/completions',
                'max_tokens': 4096,
                'cost_per_1k_tokens': {'input': 0.01, 'output': 0.03},
                'display_name': 'GPT-4 Turbo',
                'description': 'OpenAI GPT-4 Turbo - Faster and more efficient'
            },
            'gpt-3.5-turbo': {
                'provider': 'openai',
                'endpoint': 'https://api.openai.com/v1/chat/completions', 
                'max_tokens': 4096,
                'cost_per_1k_tokens': {'input': 0.0015, 'output': 0.002},
                'display_name': 'GPT-3.5 Turbo',
                'description': 'OpenAI GPT-3.5 Turbo - Fast and cost-effective'
            }
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available generic models."""
        available_models = []
        
        for model_key, model_config in self.model_configs.items():
            provider = model_config['provider']
            api_key = self.api_keys.get(provider)
            
            available_models.append({
                'key': model_key,
                'name': model_config['display_name'],
                'description': model_config['description'],
                'type': 'generic',
                'provider': provider,
                'available': api_key is not None,
                'cost_per_1k_input': model_config['cost_per_1k_tokens']['input'],
                'cost_per_1k_output': model_config['cost_per_1k_tokens']['output']
            })
        
        return available_models
    
    def generate_certificate(
        self,
        system_description: str,
        model_name: str,
        rag_context: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> ModelResponse:
        """Generate barrier certificate using a generic model."""
        
        if model_name not in self.model_configs:
            return ModelResponse(
                success=False,
                content="",
                model_name=model_name,
                error=f"Unsupported model: {model_name}"
            )
        
        model_config = self.model_configs[model_name]
        provider = model_config['provider']
        api_key = self.api_keys.get(provider)
        
        if not api_key:
            return ModelResponse(
                success=False,
                content="",
                model_name=model_name,
                error=f"API key not configured for {provider}"
            )
        
        # Create the prompt
        prompt = self._create_barrier_certificate_prompt(system_description, rag_context)
        
        try:
            if provider == 'anthropic':
                return self._call_anthropic_api(model_name, prompt, api_key, temperature, max_tokens)
            elif provider == 'openai':
                return self._call_openai_api(model_name, prompt, api_key, temperature, max_tokens)
            else:
                return ModelResponse(
                    success=False,
                    content="",
                    model_name=model_name,
                    error=f"Unsupported provider: {provider}"
                )
        
        except Exception as e:
            logger.error(f"Error calling {provider} API: {str(e)}")
            return ModelResponse(
                success=False,
                content="",
                model_name=model_name,
                error=str(e)
            )
    
    def _create_barrier_certificate_prompt(self, system_description: str, rag_context: Optional[str] = None) -> str:
        """Create a specialized prompt for barrier certificate generation."""
        
        base_prompt = f"""You are an expert in control theory and formal methods. Your task is to generate a barrier certificate for the following dynamical system.

System Description:
{system_description}

A barrier certificate is a function B(x) that proves safety by satisfying:
1. B(x) ≤ 0 for all x in the initial set
2. B(x) ≥ 0 for all x in the unsafe set  
3. dB/dt ≤ 0 (or ΔB ≤ 0 for discrete systems) in the safe region

Requirements:
- Provide the barrier function B(x) as a mathematical expression
- Use polynomial functions when possible (e.g., x₁² + x₂² - 4)
- Explain the reasoning behind your choice
- Verify that the conditions are satisfied
- If the system is discrete-time, use discrete difference instead of continuous derivative

Please provide:
1. The barrier certificate function B(x)
2. Brief verification that it satisfies the barrier conditions
3. Any assumptions about the initial set, unsafe set, or domain bounds"""

        if rag_context:
            enhanced_prompt = f"""{base_prompt}

Additional Context from Research Literature:
{rag_context}

Please use this domain knowledge to inform your barrier certificate design."""
            return enhanced_prompt
        
        return base_prompt
    
    def _call_anthropic_api(
        self, 
        model_name: str, 
        prompt: str, 
        api_key: str, 
        temperature: float, 
        max_tokens: int
    ) -> ModelResponse:
        """Call Anthropic Claude API."""
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        
        response = requests.post(
            self.model_configs[model_name]['endpoint'],
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['content'][0]['text']
            
            # Calculate token usage and cost
            usage = result.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = input_tokens + output_tokens
            
            cost_config = self.model_configs[model_name]['cost_per_1k_tokens']
            cost_estimate = (
                (input_tokens / 1000) * cost_config['input'] +
                (output_tokens / 1000) * cost_config['output']
            )
            
            return ModelResponse(
                success=True,
                content=content,
                model_name=model_name,
                tokens_used=total_tokens,
                cost_estimate=cost_estimate,
                metadata={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'provider': 'anthropic'
                }
            )
        else:
            error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
            return ModelResponse(
                success=False,
                content="",
                model_name=model_name,
                error=error_msg
            )
    
    def _call_openai_api(
        self, 
        model_name: str, 
        prompt: str, 
        api_key: str, 
        temperature: float, 
        max_tokens: int
    ) -> ModelResponse:
        """Call OpenAI GPT API."""
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        data = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        response = requests.post(
            self.model_configs[model_name]['endpoint'],
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Calculate token usage and cost
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
            
            cost_config = self.model_configs[model_name]['cost_per_1k_tokens']
            cost_estimate = (
                (prompt_tokens / 1000) * cost_config['input'] +
                (completion_tokens / 1000) * cost_config['output']
            )
            
            return ModelResponse(
                success=True,
                content=content,
                model_name=model_name,
                tokens_used=total_tokens,
                cost_estimate=cost_estimate,
                metadata={
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'provider': 'openai'
                }
            )
        else:
            error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
            return ModelResponse(
                success=False,
                content="",
                model_name=model_name,
                error=error_msg
            )
    
    def test_api_connection(self, provider: str) -> Dict[str, Any]:
        """Test API connection for a provider."""
        api_key = self.api_keys.get(provider)
        
        if not api_key:
            return {
                'success': False,
                'provider': provider,
                'error': 'API key not configured'
            }
        
        try:
            if provider == 'anthropic':
                # Simple test message to Claude
                response = self._call_anthropic_api(
                    'claude-3-haiku', 
                    'Hello, please respond with "API connection successful"',
                    api_key,
                    0.1,
                    50
                )
            elif provider == 'openai':
                # Simple test message to GPT
                response = self._call_openai_api(
                    'gpt-3.5-turbo',
                    'Hello, please respond with "API connection successful"', 
                    api_key,
                    0.1,
                    50
                )
            else:
                return {
                    'success': False,
                    'provider': provider,
                    'error': 'Unsupported provider'
                }
            
            return {
                'success': response.success,
                'provider': provider,
                'error': response.error if not response.success else None
            }
        
        except Exception as e:
            return {
                'success': False,
                'provider': provider,
                'error': str(e)
            } 