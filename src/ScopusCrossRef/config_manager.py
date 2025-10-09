"""
Configuration Manager for Neo4j Knowledge Graph Builder
WITH VECTOR STORE SUPPORT
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration class for all scripts"""
    
    # Neo4j Configuration
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    
    # API Configuration
    scopus_api_key: str
    crossref_email: str
    
    # Directory Configuration
    data_dir: str
    
    # Processing Limits
    max_papers_enrich: int
    max_papers_import: int
    max_papers_citations: int
    max_papers_authors: int
    
    # Batch Sizes
    batch_size_import: int
    batch_size_citations: int
    batch_size_authors: int
    
    # Parallel Processing
    enable_parallel_processing: bool
    
    # Vector Store Configuration
    vector_store_enabled: bool
    embedding_model: str
    vector_dimension: int
    similarity_function: str
    batch_size_embedding: int
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        return cls(
            # Neo4j
            neo4j_uri=os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD', ''),
            neo4j_database=os.getenv('NEO4J_DATABASE', 'neo4j'),
            
            # APIs
            scopus_api_key=os.getenv('SCOPUS_API_KEY', ''),
            crossref_email=os.getenv('CROSSREF_EMAIL', ''),
            
            # Directories
            data_dir=os.getenv('DATA_DIR', './data_checkpoints'),
            
            # Limits
            max_papers_enrich=int(os.getenv('MAX_PAPERS_ENRICH', '0')),
            max_papers_import=int(os.getenv('MAX_PAPERS_IMPORT', '0')),
            max_papers_citations=int(os.getenv('MAX_PAPERS_CITATIONS', '0')),
            max_papers_authors=int(os.getenv('MAX_PAPERS_AUTHORS', '0')),
            
            # Batch sizes
            batch_size_import=int(os.getenv('BATCH_SIZE_IMPORT', '50')),
            batch_size_citations=int(os.getenv('BATCH_SIZE_CITATIONS', '5')),
            batch_size_authors=int(os.getenv('BATCH_SIZE_AUTHORS', '100')),
            
            # Parallel processing
            enable_parallel_processing=os.getenv('ENABLE_PARALLEL_PROCESSING', 'true').lower() == 'true',
            
            # Vector Store
            vector_store_enabled=os.getenv('VECTOR_STORE_ENABLED', 'true').lower() == 'true',
            embedding_model=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            vector_dimension=int(os.getenv('VECTOR_DIMENSION', '384')),
            similarity_function=os.getenv('SIMILARITY_FUNCTION', 'cosine'),
            batch_size_embedding=int(os.getenv('BATCH_SIZE_EMBEDDING', '50')),
        )
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration and return status and errors"""
        errors = []
        
        if not self.neo4j_password:
            errors.append("NEO4J_PASSWORD is required")
        
        if not self.scopus_api_key:
            errors.append("SCOPUS_API_KEY is required")
        
        if not self.crossref_email or '@' not in self.crossref_email:
            errors.append("Valid CROSSREF_EMAIL is required")
        
        # Create data directory if it doesn't exist
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create data directory {self.data_dir}: {e}")
        
        # Validate vector store config if enabled
        if self.vector_store_enabled:
            if self.vector_dimension not in [384, 768, 1024, 1536]:
                errors.append(f"Invalid VECTOR_DIMENSION: {self.vector_dimension}. Common values: 384, 768, 1024, 1536")
            
            if self.similarity_function not in ['cosine', 'euclidean']:
                errors.append(f"Invalid SIMILARITY_FUNCTION: {self.similarity_function}. Use 'cosine' or 'euclidean'")
        
        return len(errors) == 0, errors

def get_config() -> Config:
    """Get validated configuration"""
    config = Config.from_env()
    is_valid, errors = config.validate()
    
    if not is_valid:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration")
    
    return config

def print_config_status():
    """Print current configuration status for debugging"""
    try:
        config = get_config()
        print("Configuration loaded successfully:")
        print(f"  - Neo4j URI: {config.neo4j_uri}")
        print(f"  - Neo4j Database: {config.neo4j_database}")
        print(f"  - Data Directory: {config.data_dir}")
        print(f"  - Scopus API Key: {'*' * len(config.scopus_api_key[:-4])}...{config.scopus_api_key[-4:] if config.scopus_api_key else 'NOT SET'}")
        print(f"  - Crossref Email: {config.crossref_email}")
        print(f"  - Parallel Processing: {config.enable_parallel_processing}")
        print(f"  - Batch Sizes: Import={config.batch_size_import}, Citations={config.batch_size_citations}, Authors={config.batch_size_authors}")
        
        # Vector Store info
        print(f"\n  Vector Store Configuration:")
        print(f"  - Enabled: {config.vector_store_enabled}")
        print(f"  - Model: {config.embedding_model}")
        print(f"  - Dimension: {config.vector_dimension}")
        print(f"  - Similarity: {config.similarity_function}")
        print(f"  - Batch Size: {config.batch_size_embedding}")
        
        return True
    except Exception as e:
        print(f"Configuration error: {e}")
        return False

def update_env_file(key: str, value: str, env_file: str = '.env'):
    """Update or add a key-value pair in the .env file"""
    import tempfile
    import shutil
    
    lines = []
    key_found = False
    
    # Read existing file if it exists
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
    # Update existing key or prepare to add new one
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break
    
    # Add new key if not found
    if not key_found:
        lines.append(f"{key}={value}\n")
    
    # Write back to file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.writelines(lines)
        tmp_name = tmp_file.name
    
    shutil.move(tmp_name, env_file)
    print(f"Updated {key} in {env_file}")

def create_default_env_file():
    """Create a default .env file with example values"""
    default_content = """# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Scopus API Configuration
SCOPUS_API_KEY=your_scopus_api_key_here

# Crossref API Configuration
CROSSREF_EMAIL=your_email@domain.com

# Data Directory
DATA_DIR=./data_checkpoints

# Processing Limits (0 = unlimited)
MAX_PAPERS_ENRICH=0
MAX_PAPERS_IMPORT=0
MAX_PAPERS_CITATIONS=0
MAX_PAPERS_AUTHORS=0

# Batch Sizes
BATCH_SIZE_IMPORT=50
BATCH_SIZE_CITATIONS=5
BATCH_SIZE_AUTHORS=100

# Enable/Disable Parallel Processing
ENABLE_PARALLEL_PROCESSING=true

# Vector Store Configuration
VECTOR_STORE_ENABLED=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
SIMILARITY_FUNCTION=cosine
BATCH_SIZE_EMBEDDING=50
"""
    
    with open('.env', 'w') as f:
        f.write(default_content)
    
    print("Created default .env file")
    print("Please edit .env with your actual credentials before running the scripts")

def check_env_file():
    """Check if .env file exists and is properly configured"""
    if not os.path.exists('.env'):
        print(".env file not found")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("Copied .env.example to .env")
        else:
            create_default_env_file()
        return False
    
    # Check for required variables
    required_vars = ['NEO4J_PASSWORD', 'SCOPUS_API_KEY', 'CROSSREF_EMAIL']
    missing_vars = []
    
    load_dotenv()
    for var in required_vars:
        value = os.getenv(var)
        if not value or value in ['your_password_here', 'your_scopus_api_key_here', 'your_email@domain.com']:
            missing_vars.append(var)
    
    if missing_vars:
        print("The following required variables need to be set in .env:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    return True

def test_configuration():
    """Test the configuration by attempting to connect to services"""
    print("Testing configuration...")
    
    try:
        config = get_config()
        print("Configuration loaded successfully")
        
        # Test Neo4j connection
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
            with driver.session(database=config.neo4j_database) as session:
                session.run("RETURN 1")
            driver.close()
            print("Neo4j connection successful")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False
        
        # Test directory creation
        try:
            os.makedirs(config.data_dir, exist_ok=True)
            print(f"Data directory accessible: {config.data_dir}")
        except Exception as e:
            print(f"Cannot create data directory: {e}")
            return False
        
        # Test Vector Store model availability (if enabled)
        if config.vector_store_enabled:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"SentenceTransformers available")
                print(f"Testing model: {config.embedding_model}")
                print(f"(Model will be downloaded on first use)")
            except ImportError:
                print(f"SentenceTransformers not installed. Install with:")
                print(f"pip install sentence-transformers")
        
        print("All configuration tests passed")
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Configuration Manager - Neo4j Knowledge Graph Builder")
    print("=" * 60)
    
    if check_env_file():
        print_config_status()
        test_configuration()
    else:
        print("\nPlease configure your .env file before running the scripts")
        print("Edit the .env file with your actual credentials and run this script again to test")