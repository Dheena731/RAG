import logging
import sys
from pathlib import Path
from typing import List
import json

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def validate_files(file_paths: List[Path], supported_formats: List[str]) -> List[Path]:
    """
    Validate that files exist and have supported formats
    
    Args:
        file_paths: List of file paths to validate
        supported_formats: List of supported file extensions
    
    Returns:
        List of valid file paths
    """
    valid_files = []
    
    for file_path in file_paths:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        if not file_path.is_file():
            print(f"⚠️  Not a file: {file_path}")
            continue
        
        if file_path.suffix.lower() not in supported_formats:
            print(f"⚠️  Unsupported format: {file_path} (supported: {supported_formats})")
            continue
        
        valid_files.append(file_path)
    
    return valid_files


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def get_file_info(file_path: Path) -> dict:
    """Get basic information about a file"""
    try:
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "path": str(file_path),
            "size": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "extension": file_path.suffix.lower(),
            "modified": stat.st_mtime
        }
    except Exception as e:
        return {"error": str(e)}


def save_search_results(results: List[dict], output_file: Path):
    """Save search results to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")


def load_search_results(input_file: Path) -> List[dict]:
    """Load search results from a JSON file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"❌ Error loading results: {str(e)}")
        return []


def print_system_info():
    """Print system information"""
    import torch
    import platform
    
    print("🖥️  System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'chromadb',
        'sentence_transformers', 
        'torch',
        'torchvision',
        'PIL',
        'PyPDF2',
        'docx',
        'numpy',
        'pandas',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'docx':
                import docx
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True


def estimate_processing_time(file_paths: List[Path]) -> dict:
    """Estimate processing time based on file sizes and types"""
    total_size = 0
    file_counts = {"pdf": 0, "docx": 0, "txt": 0, "image": 0}
    
    for file_path in file_paths:
        try:
            size = file_path.stat().st_size
            total_size += size
            
            ext = file_path.suffix.lower()
            if ext == '.pdf':
                file_counts["pdf"] += 1
            elif ext == '.docx':
                file_counts["docx"] += 1
            elif ext in ['.txt', '.md']:
                file_counts["txt"] += 1
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                file_counts["image"] += 1
        except:
            continue
    
    # Rough estimates (very approximate)
    estimated_seconds = (
        file_counts["pdf"] * 5 +
        file_counts["docx"] * 3 +
        file_counts["txt"] * 1 +
        file_counts["image"] * 2
    )
    
    return {
        "total_files": sum(file_counts.values()),
        "total_size_mb": total_size / (1024 * 1024),
        "file_counts": file_counts,
        "estimated_time_seconds": estimated_seconds,
        "estimated_time_formatted": f"{estimated_seconds // 60}m {estimated_seconds % 60}s"
    }


def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "embedding_models": {
            "text": "sentence-transformers/all-MiniLM-L6-v2",
            "multimodal": "sentence-transformers/clip-ViT-B-32"
        },
        "chunk_settings": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "vector_db": {
            "collection_name": "multimodal_documents",
            "similarity_metric": "cosine"
        },
        "supported_formats": {
            "text": [".txt", ".pdf", ".docx", ".md"],
            "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        }
    }
    
    config_file = Path("sample_config.json")
    
    try:
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"📄 Sample config created: {config_file}")
    except Exception as e:
        print(f"❌ Error creating config: {str(e)}")
'''

# 8. Create the __init__.py file
init_py = '''"""
Multimodal RAG System Package
"""

from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import Retriever
from .utils import setup_logging, validate_files, print_system_info

__version__ = "1.0.0"
__author__ = "Multimodal RAG System"

__all__ = [
    "DocumentProcessor",
    "EmbeddingGenerator", 
    "VectorStore",
    "Retriever",
    "setup_logging",
    "validate_files",
    "print_system_info"
]