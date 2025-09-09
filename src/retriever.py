"""
Retriever component for handling search queries in the multimodal RAG system
"""
from typing import List, Dict, Any, Optional
import logging
from .response_generator import ResponseGenerator, LocalLLMGenerator
import numpy as np

from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)




class Retriever:
    def __init__(self, config):
        self.config = config
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_store = VectorStore(config)
        self.vector_store.initialize()
    
    def search(self, query: str, top_k: int = 5, 
               content_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on a text query
        
        Args:
            query: Search query string
            top_k: Number of results to return
            content_type_filter: Filter by content type ("text", "image", etc.)
        
        Returns:
            List of relevant documents with metadata and scores
        """
        try:
            logger.info(f"Searching for: '{query}' (top_k={top_k})")
            
            # Generate query embedding using the multimodal model
            # This allows the query to match both text and image content
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            # Prepare metadata filter if specified
            filter_metadata = {}
            if content_type_filter:
                filter_metadata["type"] = content_type_filter
            
            # Search in vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata if filter_metadata else None
            )
            
            # Post-process results
            processed_results = self._post_process_results(results, query)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise
    
    def search_similar_to_document(self, document_content: str, content_type: str = "text",
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document
        
        Args:
            document_content: Content of the reference document
            content_type: Type of content ("text" or "image")
            top_k: Number of results to return
        
        Returns:
            List of similar documents
        """
        try:
            logger.info(f"Finding documents similar to provided {content_type} content")
            
            # Generate embedding for the reference document
            if content_type == "text":
                doc_embedding = self.embedding_generator.generate_text_embedding(document_content)
            elif content_type == "image":
                doc_embedding = self.embedding_generator.generate_image_embedding(document_content)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Search for similar documents
            results = self.vector_store.search(
                query_embedding=doc_embedding,
                top_k=top_k + 1  # +1 to account for the document itself if it's in the store
            )
            
            # Filter out the exact same document if present
            filtered_results = []
            for result in results:
                if result["score"] < 0.99:  # Avoid exact matches
                    filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def hybrid_search(self, text_query: str, image_query: str = None,
                     text_weight: float = 0.7, image_weight: float = 0.3,
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both text and image queries
        
        Args:
            text_query: Text query string
            image_query: Base64 encoded image or image path
            text_weight: Weight for text similarity
            image_weight: Weight for image similarity
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with combined scores
        """
        try:
            logger.info(f"Performing hybrid search with text and image")
            
            # Generate embeddings
            text_embedding = self.embedding_generator.generate_text_embedding(text_query)
            
            if image_query:
                image_embedding = self.embedding_generator.generate_image_embedding(image_query)
                combined_embedding = (
                    text_weight * text_embedding + 
                    image_weight * image_embedding
                )
                
                # Normalize the combined embedding
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            else:
                combined_embedding = text_embedding
            
            # Search using combined embedding
            results = self.vector_store.search(
                query_embedding=combined_embedding,
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filter
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            top_k: Maximum number of results to return
        
        Returns:
            List of documents matching the metadata filter
        """
        try:
            logger.info(f"Searching by metadata filter: {metadata_filter}")
            
            # Use a dummy query embedding (won't be used for scoring when we have metadata filter)
            dummy_embedding = self.embedding_generator.generate_text_embedding("dummy query")
            
            results = self.vector_store.search(
                query_embedding=dummy_embedding,
                top_k=top_k,
                filter_metadata=metadata_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            raise
    
    def get_document_by_source(self, source_path: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get all documents from a specific source file
        
        Args:
            source_path: Path to the source file
            top_k: Maximum number of results to return
        
        Returns:
            List of documents from the specified source
        """
        return self.search_by_metadata({"source": source_path}, top_k=top_k)
    
    def _post_process_results(self, results: List[Dict[str, Any]], 
                            original_query: str) -> List[Dict[str, Any]]:
        """
        Post-process search results to add additional information and improve relevance
        
        Args:
            results: Raw search results from vector store
            original_query: Original search query
        
        Returns:
            Post-processed results
        """
        try:
            processed_results = []
            
            for result in results:
                processed_result = result.copy()
                
                # Add relevance explanation for text content
                if result.get("type") == "text":
                    processed_result["relevance_explanation"] = self._generate_relevance_explanation(
                        result["content"], original_query
                    )
                
                # Add content preview for long text
                content = result.get("content", "")
                if len(content) > 500:
                    processed_result["content_preview"] = content[:500] + "..."
                    processed_result["full_content_available"] = True
                else:
                    processed_result["content_preview"] = content
                    processed_result["full_content_available"] = False
                
                # Add formatted metadata
                processed_result["formatted_metadata"] = self._format_metadata(result.get("metadata", {}))
                
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            logger.warning(f"Error in post-processing results: {str(e)}")
            return results  # Return original results if post-processing fails
    
    def _generate_relevance_explanation(self, content: str, query: str) -> str:
        """
        Generate a simple explanation of why a document is relevant
        This is a basic implementation - could be enhanced with more sophisticated NLP
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        common_words = query_words.intersection(content_words)
        
        if common_words:
            return f"Contains keywords: {', '.join(list(common_words)[:3])}"
        else:
            return "Semantically similar content"
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for display"""
        formatted_parts = []
        
        # Priority fields to show first
        priority_fields = ["source", "file_type", "page", "chunk"]
        
        for field in priority_fields:
            if field in metadata:
                value = metadata[field]
                if field == "source":
                    # Show just the filename, not the full path
                    import os
                    value = os.path.basename(str(value))
                formatted_parts.append(f"{field}: {value}")
        
        # Add other interesting fields
        for key, value in metadata.items():
            if key not in priority_fields and key not in ["doc_id", "type"]:
                if len(str(value)) < 50:  # Only show short values
                    formatted_parts.append(f"{key}: {value}")
        
        return " | ".join(formatted_parts)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        return self.vector_store.get_collection_info()
class EnhancedRetriever(Retriever):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize generator
        if config.USE_LOCAL_LLM:
            self.generator = LocalLLMGenerator(config)
        else:
            self.generator = ResponseGenerator(config)
    
    def search_and_generate(self, query: str, top_k: int = 5, 
                          generate_response: bool = True) -> Dict[str, Any]:
        """
        Search documents and optionally generate LLM response
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            generate_response: Whether to generate LLM response
            
        Returns:
            Dict with search results and optional generated response
        """
        # Get search results
        search_results = self.search(query, top_k=top_k)
        
        result = {
            "query": query,
            "search_results": search_results,
            "num_results": len(search_results)
        }
        
        # Generate response if requested
        if generate_response and search_results:
            # Limit context to avoid token limits
            context_docs = search_results[:self.config.MAX_CONTEXT_DOCS]
            
            generated = self.generator.generate_response(query, context_docs)
            result.update(generated)
        
        return result