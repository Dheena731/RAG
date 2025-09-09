"""
Response generator using OpenRouter API for LLM generation
"""
import requests
import json
from typing import List, Dict, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class ResponseGenerator:
    def __init__(self, config):
        self.config = config
        self.openrouter_api_key = OPENROUTER_API_KEY
        self.model = config.LLM_MODEL
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response using retrieved documents and LLM
        
        Args:
            query: User's original query
            retrieved_docs: List of retrieved documents from vector search
            
        Returns:
            Dict containing generated response and metadata
        """
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Create prompt
            prompt = self._create_prompt(query, context, retrieved_docs)
            
            # Generate response using OpenRouter
            response = self._call_openrouter(prompt)
            
            # Format final response
            formatted_response = {
                "answer": response,
                "query": query,
                "sources": self._extract_sources(retrieved_docs),
                "context_used": len(retrieved_docs),
                "retrieval_scores": [doc.get("score", 0) for doc in retrieved_docs]
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"I encountered an error while generating a response: {str(e)}",
                "query": query,
                "sources": [],
                "error": True
            }
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            doc_type = doc.get("type", "Unknown")
            
            # Handle image documents
            if doc_type == "image":
                ocr_text = doc.get("metadata", {}).get("ocr_text", "")
                if ocr_text:
                    content = f"Image content (OCR): {ocr_text}"
                else:
                    content = f"Image from {source} (visual content)"
            
            context_parts.append(f"[Source {i}] {source}:\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create a detailed prompt for the LLM"""
        
        prompt = f"""You are an AI assistant helping users find information from their document collection. Use the provided context to answer the user's question accurately and comprehensively.

USER QUESTION: {query}

RELEVANT DOCUMENTS:
{context}

INSTRUCTIONS:
1. Answer the user's question based on the provided documents
2. Synthesize information from multiple sources when relevant
3. If the documents contain images, acknowledge visual content appropriately
4. Be specific and cite which sources you're referencing
5. If the provided documents don't fully answer the question, acknowledge this
6. Maintain accuracy - don't make up information not present in the sources

Please provide a clear, well-structured response:"""

        return prompt
    
    def _call_openrouter(self, prompt: str) -> str:
        """Make API call to OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://localhost",  # Optional
            "X-Title": "Multimodal RAG System",  # Optional
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing OpenRouter response: {str(e)}")
            raise
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from retrieved documents"""
        sources = []
        seen_sources = set()
        
        for doc in retrieved_docs:
            source = doc.get("source", "Unknown")
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)
        
        return sources


class LocalLLMGenerator:
    """Alternative generator using local LLM (e.g., Ollama, llamafile)"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = config.LOCAL_LLM_URL  # e.g., "http://localhost:11434"
        self.model = config.LOCAL_LLM_MODEL   # e.g., "llama2", "mistral"
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using local LLM"""
        try:
            context = self._prepare_context(retrieved_docs)
            prompt = self._create_prompt(query, context)
            
            # Call local LLM API (Ollama format)
            response = self._call_local_llm(prompt)
            
            return {
                "answer": response,
                "query": query,
                "sources": [doc.get("source", "Unknown") for doc in retrieved_docs],
                "context_used": len(retrieved_docs),
            }
            
        except Exception as e:
            logger.error(f"Local LLM error: {str(e)}")
            return {
                "answer": f"Error with local LLM: {str(e)}",
                "query": query,
                "sources": [],
                "error": True
            }
    
    def _call_local_llm(self, prompt: str) -> str:
        """Call local LLM API (Ollama format)"""
        import requests
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data,
            timeout=60
        )
        
        response.raise_for_status()
        return response.json()["response"]
    
    def _prepare_context(self, retrieved_docs):
        """Same as ResponseGenerator._prepare_context"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            context_parts.append(f"[Source {i}] {source}:\n{content}")
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query, context):
        """Create prompt for local LLM"""
        return f"""Based on the following documents, answer the user's question:


Question: {query}

Documents:
{context}

Answer:"""