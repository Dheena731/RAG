''' Embedding generator for creating embeddings from text and image content
Uses sentence-transformers for both text and multimodal (CLIP) embeddings'''


import base64
import io
from typing import List, Union
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        
        # Initialize embedding models
        logger.info(f"Loading embedding models on device: {self.device}")
        
        # Text embedding model(MiniLM)
        self.text_model = SentenceTransformer(
            config.TEXT_EMBEDDING_MODEL,
            device=str(self.device)
        )
        
        # Multimodal embedding model (CLIP)
        self.multimodal_model = SentenceTransformer(
            config.MULTIMODAL_EMBEDDING_MODEL,
            device=str(self.device)
        )
        
        logger.info("Embedding models loaded successfully")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text content"""
        try:
            if not text or not text.strip():
                raise ValueError("Empty text provided")
            
            # Generate embedding
            embedding = self.text_model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise
    
    def generate_image_embedding(self, image_data: Union[str, Image.Image]) -> np.ndarray:
        """Generate embedding for image content"""
        try:
            # Handle different input types
            if isinstance(image_data, str):
                # Assume it's base64 encoded
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate embedding using CLIP
            embedding = self.multimodal_model.encode(
                image,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            raise
    
    def generate_multimodal_embedding(self, text: str, image_data: Union[str, Image.Image] = None) -> np.ndarray:
        """Generate combined embedding for text and image"""
        try:
            embeddings = []
            
            # Generate text embedding
            if text and text.strip():
                text_embedding = self.generate_text_embedding(text)
                embeddings.append(text_embedding)
            
            # Generate image embedding
            if image_data is not None:
                image_embedding = self.generate_image_embedding(image_data)
                embeddings.append(image_embedding)
            
            if not embeddings:
                raise ValueError("No valid content provided for embedding")
            
            # If we have both text and image, combine them
            if len(embeddings) == 2:
                # Simple concatenation (you could also try averaging, weighted combination, etc.)
                combined_embedding = np.concatenate(embeddings)
                # Normalize the combined embedding
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
                return combined_embedding
            else:
                # Return the single embedding we have
                return embeddings[0]
                
        except Exception as e:
            logger.error(f"Error generating multimodal embedding: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        try:
            # For queries, we use the multimodal model to enable cross-modal search
            # (text query can match both text and image content)
            embedding = self.multimodal_model.encode(
                query,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def batch_generate_embeddings(self, contents: List[dict]) -> List[np.ndarray]:
        """Generate embeddings for multiple contents efficiently"""
        embeddings = []
        
        # Separate text and image contents for batch processing
        text_contents = []
        image_contents = []
        content_types = []
        
        for content in contents:
            if content["type"] == "text":
                text_contents.append(content["content"])
                content_types.append("text")
            elif content["type"] == "image":
                image_contents.append(content["content"])
                content_types.append("image")
        
        # Batch process text embeddings
        if text_contents:
            try:
                text_embeddings = self.text_model.encode(
                    text_contents,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    batch_size=32,  # Adjust based on your memory constraints
                    show_progress_bar=True
                )
                text_embeddings = [emb.astype(np.float32) for emb in text_embeddings]
            except Exception as e:
                logger.error(f"Error in batch text embedding: {str(e)}")
                # Fallback to individual processing
                text_embeddings = []
                for text in text_contents:
                    text_embeddings.append(self.generate_text_embedding(text))
        
        # Process image embeddings (CLIP doesn't support easy batching with PIL images)
        image_embeddings = []
        if image_contents:
            for image_data in image_contents:
                try:
                    image_embeddings.append(self.generate_image_embedding(image_data))
                except Exception as e:
                    logger.error(f"Error processing image in batch: {str(e)}")
                    # Create a zero embedding as fallback
                    zero_embedding = np.zeros(self.get_embedding_dimension("image"), dtype=np.float32)
                    image_embeddings.append(zero_embedding)
        
        # Combine embeddings in original order
        text_idx = 0
        image_idx = 0
        
        for content_type in content_types:
            if content_type == "text":
                embeddings.append(text_embeddings[text_idx])
                text_idx += 1
            elif content_type == "image":
                embeddings.append(image_embeddings[image_idx])
                image_idx += 1
        
        return embeddings
    
    def get_embedding_dimension(self, content_type: str = "text") -> int:
        """Get the dimension of embeddings for a specific content type"""
        if content_type == "text":
            return self.text_model.get_sentence_embedding_dimension()
        elif content_type in ["image", "multimodal"]:
            return self.multimodal_model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"Unknown content type: {content_type}")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0