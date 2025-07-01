"""
File System Connector

Monitors file systems and cloud storage for new document uploads.
Supports various document formats and real-time processing.
"""

import logging
import os
import time
import hashlib
from typing import Dict, Any, List, Generator, Optional
from datetime import datetime
from pathlib import Path
import mimetypes
from .base_connector import BaseConnector
from config import CONFIG

logger = logging.getLogger(__name__)

class FileConnector(BaseConnector):
    """
    File system connector for monitoring document uploads.
    
    Features:
    - Real-time file system monitoring
    - Support for multiple document formats
    - File metadata extraction
    - Duplicate detection
    - Batch processing for efficiency
    """
    
    def __init__(self):
        super().__init__("file_connector", CONFIG.kafka.topics["raw_documents"])
        
        self.watch_directories = CONFIG.data_sources.watch_directories
        self.poll_interval = 10  # Check every 10 seconds
        
        # Track processed files
        self.processed_files = set()
        
        # Supported file types
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.rtf',
            '.csv', '.json', '.xml', '.html', '.htm',
            '.md', '.rst', '.odt', '.pptx', '.ppt'
        }
        
        # Ensure watch directories exist
        self._ensure_directories()
        
        logger.info(f"File connector initialized for {len(self.watch_directories)} directories")
    
    def _ensure_directories(self):
        """Ensure all watch directories exist."""
        for directory in self.watch_directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.info(f"Monitoring directory: {directory}")
            except Exception as e:
                logger.error(f"Could not create/access directory {directory}: {e}")
    
    def _get_file_id(self, file_path: str, stat_info: os.stat_result) -> str:
        """
        Generate unique ID for file based on path and modification time.
        
        Args:
            file_path: Path to the file
            stat_info: File stat information
            
        Returns:
            Unique identifier for the file
        """
        content = f"{file_path}:{stat_info.st_mtime}:{stat_info.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            path_obj = Path(file_path)
            stat_info = path_obj.stat()
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            
            metadata = {
                'filename': path_obj.name,
                'file_path': str(path_obj.absolute()),
                'file_size': stat_info.st_size,
                'file_extension': path_obj.suffix.lower(),
                'mime_type': mime_type,
                'created_time': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                'accessed_time': datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return {}
    
    def _read_file_content(self, file_path: str, max_size: int = 10 * 1024 * 1024) -> Optional[str]:
        """
        Read file content with size limit.
        
        Args:
            file_path: Path to the file
            max_size: Maximum file size to read (default 10MB)
            
        Returns:
            File content as string or None if error
        """
        try:
            path_obj = Path(file_path)
            
            # Check file size
            if path_obj.stat().st_size > max_size:
                logger.warning(f"File {file_path} too large ({path_obj.stat().st_size} bytes), skipping content read")
                return None
            
            # Read based on file type
            extension = path_obj.suffix.lower()
            
            if extension in ['.txt', '.md', '.rst', '.csv', '.json', '.xml', '.html', '.htm']:
                # Text files - try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                
                logger.warning(f"Could not decode text file {file_path}")
                return None
            
            elif extension == '.pdf':
                # PDF files - basic text extraction
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    logger.warning("PyPDF2 not available for PDF processing")
                    return None
                except Exception as e:
                    logger.error(f"Error reading PDF {file_path}: {e}")
                    return None
            
            elif extension in ['.docx', '.doc']:
                # Word documents - basic text extraction
                try:
                    if extension == '.docx':
                        import docx
                        doc = docx.Document(file_path)
                        text = ""
                        for paragraph in doc.paragraphs:
                            text += paragraph.text + "\n"
                        return text
                    else:
                        logger.warning(f"Legacy .doc format not supported: {file_path}")
                        return None
                except ImportError:
                    logger.warning("python-docx not available for DOCX processing")
                    return None
                except Exception as e:
                    logger.error(f"Error reading DOCX {file_path}: {e}")
                    return None
            
            else:
                logger.debug(f"Unsupported file type for content extraction: {extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading file content {file_path}: {e}")
            return None
    
    def _scan_directory(self, directory: str) -> Generator[Dict[str, Any], None, None]:
        """
        Scan directory for new files.
        
        Args:
            directory: Directory path to scan
            
        Yields:
            Dictionary containing file data
        """
        try:
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    if not self.is_running:
                        break
                    
                    file_path = os.path.join(root, filename)
                    path_obj = Path(file_path)
                    
                    # Check if file extension is supported
                    if path_obj.suffix.lower() not in self.supported_extensions:
                        continue
                    
                    try:
                        stat_info = path_obj.stat()
                        file_id = self._get_file_id(file_path, stat_info)
                        
                        # Check if already processed
                        if file_id in self.processed_files:
                            continue
                        
                        # Check if file is still being written (size changing)
                        time.sleep(0.1)  # Small delay
                        new_stat = path_obj.stat()
                        if new_stat.st_mtime != stat_info.st_mtime or new_stat.st_size != stat_info.st_size:
                            continue  # File still being written
                        
                        # Extract metadata
                        metadata = self._extract_file_metadata(file_path)
                        if not metadata:
                            continue
                        
                        # Read content
                        content = self._read_file_content(file_path)
                        
                        # Create file data
                        file_data = {
                            'id': file_id,
                            'metadata': metadata,
                            'content': content,
                            'content_available': content is not None,
                            'source_type': 'file_system',
                            'source_directory': directory,
                            'collected_at': datetime.utcnow().isoformat()
                        }
                        
                        # Mark as processed
                        self.processed_files.add(file_id)
                        
                        # Cache processed file ID
                        self.cache_data(f"processed_file:{file_id}", True, ttl=86400)
                        
                        yield file_data
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
    
    def collect_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        Collect data from all monitored directories.
        
        Yields:
            Dictionary containing file data
        """
        for directory in self.watch_directories:
            if not self.is_running:
                break
            
            try:
                for file_data in self._scan_directory(directory):
                    yield file_data
                    
            except Exception as e:
                logger.error(f"Error collecting data from {directory}: {e}")
                self.error_count += 1
    
    def _load_processed_files(self):
        """Load previously processed file IDs from cache."""
        try:
            logger.info("Loading processed files from cache")
            # For now, we'll start fresh each time
            self.processed_files = set()
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
    
    def run(self):
        """
        Main execution loop for file connector.
        
        Continuously monitors directories for new files.
        """
        self.start()
        self._load_processed_files()
        
        logger.info(f"Starting file connector monitoring {len(self.watch_directories)} directories")
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Collect and publish data
                for file_data in self.collect_data():
                    if not self.is_running:
                        break
                    
                    # Publish to Kafka
                    success = self.publish_message(file_data, key=file_data['id'])
                    if success:
                        filename = file_data['metadata'].get('filename', 'unknown')
                        logger.info(f"Published file: {filename}")
                
                # Calculate sleep time
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed_time)
                
                if sleep_time > 0 and self.is_running:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("File connector interrupted by user")
        except Exception as e:
            logger.error(f"Error in file connector main loop: {e}")
        finally:
            self.stop()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run file connector
    connector = FileConnector()
    connector.run()

