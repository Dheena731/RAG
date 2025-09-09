
"""
Vector store implementation using ChromaDB for storing and retrieving embeddings
"""
import uuid
from typing import List, Dict, Any, Optional
import logging

import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.collection = None
        
    def initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.config.CHROMA_DB_PATH),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB client initialized at: {self.config.CHROMA_DB_PATH}")
            
            # Create or get collection
            try:
                # Try to get existing collection
                self.collection = self.client.get_collection(
                    name=self.config.COLLECTION_NAME
                )
                logger.info(f"Using existing collection: {self.config.COLLECTION_NAME}")
            except Exception as e:
                # Collection doesn't exist or other error, create it
                logger.warning(f"Collection not found or error ({e}), creating new collection.")
                self.collection = self.client.create_collection(
                    name=self.config.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.info(f"Created new collection: {self.config.COLLECTION_NAME}")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Add documents and their embeddings to the vector store"""
        try:
            if not documents or not embeddings:
                raise ValueError("Documents and embeddings cannot be empty")
            
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            # Prepare data for ChromaDB
            ids = []
            documents_content = []
            metadatas = []
            embeddings_list = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # Prepare document content for storage
                if doc["type"] == "image":
                    # For images, we store a description or the OCR text in the document field
                    ocr_text = doc["metadata"].get("ocr_text", "")
                    content = f"Image from {doc['metadata']['source']}: {ocr_text}" if ocr_text else f"Image from {doc['metadata']['source']}"
                else:
                    content = doc["content"]
                
                documents_content.append(content)
                
                # Prepare metadata
                metadata = doc["metadata"].copy()
                metadata["type"] = doc["type"]
                metadata["doc_id"] = doc_id
                
                # ChromaDB has limitations on metadata values, convert all to strings
                for key, value in metadata.items():
                    if isinstance(value, (list, dict)):
                        metadata[key] = str(value)
                    elif isinstance(value, (int, float)):
                        metadata[key] = str(value)
                    elif value is None:
                        metadata[key] = "null"
                
                metadatas.append(metadata)
                embeddings_list.append(embedding.tolist())
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents_content,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding"""
        try:
            if self.collection is None:
                raise ValueError("Vector store not initialized")
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    where_clause[key] = str(value)  # Convert to string for ChromaDB
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity score
                        "distance": results["distances"][0][i]
                    }
                    
                    # Add derived fields
                    result["source"] = result["metadata"].get("source", "Unknown")
                    result["type"] = result["metadata"].get("type", "Unknown")
                    
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if self.collection is None:
                raise ValueError("Vector store not initialized")
            
            count = self.collection.count()
            
            # Get a sample of documents to understand the structure
            sample = self.collection.peek(limit=5)
            
            return {
                "name": self.config.COLLECTION_NAME,
                "count": count,
                "sample_metadata_keys": list(sample["metadatas"][0].keys()) if sample["metadatas"] else [],
                "sample_document_types": list(set(
                    meta.get("type", "unknown") for meta in sample["metadatas"]
                )) if sample["metadatas"] else []
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def delete_documents(self, filter_metadata: Dict[str, Any]):
        """Delete documents based on metadata filter"""
        try:
            if self.collection is None:
                raise ValueError("Vector store not initialized")
            
            # Convert filter values to strings
            where_clause = {}
            for key, value in filter_metadata.items():
                where_clause[key] = str(value)
            
            # Delete documents
            self.collection.delete(where=where_clause)
            
            logger.info(f"Deleted documents matching filter: {filter_metadata}")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def reset_collection(self):
        """Reset (clear) the entire collection"""
        try:
            if self.client is None:
                raise ValueError("ChromaDB client not initialized")
            
            # Delete existing collection
            try:
                self.client.delete_collection(name=self.config.COLLECTION_NAME)
            except ValueError:
                pass  # Collection doesn't exist
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Reset collection: {self.config.COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise
    
    def update_document(self, doc_id: str, content: str = None, 
                       metadata: Dict[str, Any] = None, embedding: np.ndarray = None):
        """Update a specific document"""
        try:
            if self.collection is None:
                raise ValueError("Vector store not initialized")
            
            update_data = {"ids": [doc_id]}
            
            if content is not None:
                update_data["documents"] = [content]
            
            if metadata is not None:
                # Convert metadata values to strings
                processed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (list, dict)):
                        processed_metadata[key] = str(value)
                    elif isinstance(value, (int, float)):
                        processed_metadata[key] = str(value)
                    elif value is None:
                        processed_metadata[key] = "null"
                    else:
                        processed_metadata[key] = str(value)
                
                update_data["metadatas"] = [processed_metadata]
            
            if embedding is not None:
                update_data["embeddings"] = [embedding.tolist()]
            
            self.collection.update(**update_data)
            
            logger.info(f"Updated document: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            raise
    
    def close(self):
        """Close the vector store connection"""
        try:
            # ChromaDB PersistentClient automatically persists data
            # No explicit close needed, but we can clean up references
            self.collection = None
            self.client = None
            
            logger.info("Vector store connection closed")
            
        except Exception as e:
            logger.error(f"Error closing vector store: {str(e)}")
