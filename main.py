''' Main CLI entry point for the multimodal RAG system
Usage: python main.py --command [ingest|query] --files [file1.pdf file2.jpg ...] --query "your query" '''

import argparse
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

from config import Config
from src.document_processor import DocumentProcessor
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import EnhancedRetriever, Retriever
from src.utils import setup_logging, validate_files


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG System CLI")
    parser.add_argument(
        "--command", 
        choices=["ingest", "query", "setup"], 
        required=True,
        help="Command to execute: ingest (process documents), query (search), or setup (initialize system)"
    )
    parser.add_argument(
        "--files", 
        nargs="*", 
        help="List of files to ingest (supports PDF, DOCX, TXT, images)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing files to ingest"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Query text for searching documents"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=10,
        help="Number of top results to return (default: 5)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
    "--generate", 
    action="store_true",
    help="Generate LLM response (requires OpenRouter API key or local LLM)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Initialize configuration
    config = Config()
    
    if args.command == "setup":
        print("🚀 Setting up Multimodal RAG System...")
        setup_system(config)
        return
        
    elif args.command == "ingest":
        if not args.files and not args.directory:
            print("❌ Error: Please provide either --files or --directory for ingestion")
            sys.exit(1)
        
        files_to_process = []
        
        if args.files:
            files_to_process.extend([Path(f) for f in args.files])
        
        if args.directory:
            directory = Path(args.directory)
            if directory.exists() and directory.is_dir():
                # Get all supported files from directory
                for ext in config.SUPPORTED_TEXT_FORMATS + config.SUPPORTED_IMAGE_FORMATS:
                    files_to_process.extend(directory.glob(f"**/*{ext}"))
        
        if not files_to_process:
            print("❌ No supported files found")
            sys.exit(1)
            
        print(f"📂 Processing {len(files_to_process)} files...")
        ingest_documents(config, files_to_process)
        
    elif args.command == "query":
        if not args.query:
            print("❌ Error: Please provide a query using --query")
            sys.exit(1)
        
    print(f"🔍 Searching for: '{args.query}'")
    results = search_documents(config, args.query, args.top_k, generate=args.generate)
    print("\nDEBUG RAW RESULTS:\n", results)
    display_results(results)

def setup_system(config):
    """Initialize the RAG system"""
    try:
        print("📦 Initializing vector database...")
        vector_store = VectorStore(config)
        vector_store.initialize()
        print("✅ System setup complete!")
        
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        sys.exit(1)


def ingest_documents(config, file_paths: List[Path]):
    """Process and ingest documents into the vector database"""
    try:
        # Initialize components
        doc_processor = DocumentProcessor(config)
        embedding_generator = EmbeddingGenerator(config)
        vector_store = VectorStore(config)
        
        print("🔧 Initializing vector database...")
        vector_store.initialize()
        
        processed_count = 0
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                if not file_path.exists():
                    print(f"⚠️  File not found: {file_path}")
                    continue
                
                print(f"📄 Processing: {file_path.name}")
                
                # Process document based on type
                file_type = doc_processor.detect_file_type(file_path)
                chunks = doc_processor.process_file(file_path, file_type)
                
                if not chunks:
                    print(f"⚠️  No content extracted from: {file_path.name}")
                    continue
                
                print(f"📝 Extracted {len(chunks)} chunks")
                
                # Generate embeddings
                embeddings = []
                for chunk in chunks:
                    if file_type == "image":
                        embedding = embedding_generator.generate_image_embedding(chunk["content"])
                    else:
                        embedding = embedding_generator.generate_text_embedding(chunk["content"])
                    embeddings.append(embedding)
                
                # Store in vector database
                vector_store.add_documents(chunks, embeddings)
                processed_count += 1
                print(f"✅ Stored {len(chunks)} chunks from {file_path.name}")
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                continue
        
        print(f"🎉 Successfully processed {processed_count} files!")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")
        sys.exit(1)

def search_documents(config, query: str, top_k: int, generate: bool = False):
    """Search documents and optionally generate response"""
    try:
        if generate:
            retriever = EnhancedRetriever(config)
            results = retriever.search_and_generate(query, top_k=top_k, generate_response=True)
        else:
            retriever = Retriever(config) 
            results = retriever.search(query, top_k=top_k)
        
        return results
        
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        sys.exit(1)


def display_search_results(results):
    """Display search results in a formatted way"""
    if not results:
        print("🔍 No results found")
        return
    
    print(f"\\n🎯 Found {len(results)} relevant results:")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\\n{i}. Score: {result.get('score', 0):.4f}")
        print(f"   Source: {result.get('source', 'Unknown')}")
        print(f"   Type: {result.get('type', 'Unknown')}")
        
        content = result.get('content', '')
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"   Content: {content}")
        
        if result.get('metadata'):
            print(f"   Metadata: {result['metadata']}")
def display_results(results):
    """Display search results and optional generated response"""
    if isinstance(results, dict) and "answer" in results:
        # Enhanced response with generation
        print(f"Response:")
        print("=" * 50)
        print(results["answer"])
        
        print(f"\n📚 Sources Used ({len(results['sources'])}):")
        for i, source in enumerate(results["sources"], 1):
            print(f"  {i}. {source}")
        
        print(f"\n🔍 Retrieved {results['num_results']} documents")
        
        # Optionally show search results details
        if input("\nShow detailed search results? (y/N): ").lower() == 'y':
            display_search_results(results["search_results"])
    
    else:
        # Regular search results
        display_search_results(results)

if __name__ == "__main__":
    main()
