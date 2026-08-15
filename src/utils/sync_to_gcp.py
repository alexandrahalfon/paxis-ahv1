#!/usr/bin/env python3
"""
Sync Processed Documents to GCP Bucket

Uploads processed documents to Google Cloud Storage bucket.
Can sync entire folder or individual documents.
"""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


class GCPBucketSync:
    """Sync processed documents to GCP bucket."""

    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize GCP sync.
        
        Args:
            bucket_name: GCP bucket name (if None, uses environment variable)
        """
        from google.cloud import storage
        
        # Get bucket name
        self.bucket_name = bucket_name or os.getenv('GCP_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("GCP_BUCKET_NAME not set. Please set in .env file or pass as argument.")
        
        # Initialize GCP client
        # Assumes GOOGLE_APPLICATION_CREDENTIALS is set in environment
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        
        print(f"✓ Connected to GCP bucket: {self.bucket_name}")

    def sync_folder(self, local_folder: str, gcp_prefix: str = "processed_documents") -> dict:
        """
        Sync entire local folder to GCP bucket.
        
        Args:
            local_folder: Local folder path (e.g., "processed_documents")
            gcp_prefix: Prefix in GCP bucket (folder path in bucket)
            
        Returns:
            Dictionary with sync statistics
        """
        local_path = Path(local_folder)
        
        if not local_path.exists():
            raise ValueError(f"Local folder not found: {local_folder}")
        
        print(f"\n{'='*70}")
        print(f"SYNCING TO GCP BUCKET")
        print(f"{'='*70}")
        print(f"Local folder: {local_path}")
        print(f"GCP bucket: gs://{self.bucket_name}/{gcp_prefix}")
        print(f"{'='*70}\n")
        
        stats = {
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
            "total_size_mb": 0
        }
        
        # Get all files recursively
        all_files = list(local_path.rglob("*"))
        file_count = len([f for f in all_files if f.is_file()])
        
        print(f"Found {file_count} files to sync\n")
        
        for local_file in all_files:
            if local_file.is_file():
                # Calculate relative path
                relative_path = local_file.relative_to(local_path)
                gcp_path = f"{gcp_prefix}/{relative_path}".replace("\\", "/")
                
                # Upload file
                try:
                    blob = self.bucket.blob(gcp_path)
                    
                    # Check if file already exists and is same size
                    if blob.exists():
                        blob.reload()
                        local_size = local_file.stat().st_size
                        if blob.size == local_size:
                            print(f"⏭️  Skipped (unchanged): {relative_path}")
                            stats["skipped"] += 1
                            continue
                    
                    # Upload
                    blob.upload_from_filename(str(local_file))
                    file_size_mb = local_file.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += file_size_mb
                    stats["uploaded"] += 1
                    
                    print(f"✓ Uploaded: {relative_path} ({file_size_mb:.2f} MB)")
                    
                except Exception as e:
                    print(f"✗ Failed: {relative_path} - {e}")
                    stats["failed"] += 1
        
        print(f"\n{'='*70}")
        print(f"SYNC COMPLETE")
        print(f"{'='*70}")
        print(f"Uploaded: {stats['uploaded']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        print(f"{'='*70}\n")
        
        return stats

    def sync_document(self, doc_name: str, local_base: str = "processed_documents",
                     gcp_prefix: str = "processed_documents") -> dict:
        """
        Sync a single document folder to GCP.
        
        Args:
            doc_name: Document name (folder name)
            local_base: Base local folder
            gcp_prefix: Prefix in GCP bucket
            
        Returns:
            Dictionary with sync statistics
        """
        local_folder = Path(local_base) / doc_name
        
        if not local_folder.exists():
            raise ValueError(f"Document folder not found: {local_folder}")
        
        print(f"\nSyncing document: {doc_name}")
        
        stats = {
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
            "total_size_mb": 0
        }
        
        for local_file in local_folder.rglob("*"):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_folder)
                gcp_path = f"{gcp_prefix}/{doc_name}/{relative_path}".replace("\\", "/")
                
                try:
                    blob = self.bucket.blob(gcp_path)
                    
                    # Check if exists
                    if blob.exists():
                        blob.reload()
                        local_size = local_file.stat().st_size
                        if blob.size == local_size:
                            stats["skipped"] += 1
                            continue
                    
                    # Upload
                    blob.upload_from_filename(str(local_file))
                    file_size_mb = local_file.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += file_size_mb
                    stats["uploaded"] += 1
                    
                    print(f"  ✓ {relative_path}")
                    
                except Exception as e:
                    print(f"  ✗ {relative_path} - {e}")
                    stats["failed"] += 1
        
        print(f"  Uploaded: {stats['uploaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
        
        return stats

    def download_folder(self, gcp_prefix: str, local_folder: str) -> dict:
        """
        Download entire folder from GCP bucket to local.
        
        Args:
            gcp_prefix: Prefix in GCP bucket
            local_folder: Local destination folder
            
        Returns:
            Dictionary with download statistics
        """
        local_path = Path(local_folder)
        local_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"DOWNLOADING FROM GCP BUCKET")
        print(f"{'='*70}")
        print(f"GCP bucket: gs://{self.bucket_name}/{gcp_prefix}")
        print(f"Local folder: {local_path}")
        print(f"{'='*70}\n")
        
        stats = {
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "total_size_mb": 0
        }
        
        # List all blobs with prefix
        blobs = self.bucket.list_blobs(prefix=gcp_prefix)
        
        for blob in blobs:
            # Skip if it's a "folder" (ends with /)
            if blob.name.endswith('/'):
                continue
            
            # Calculate local path
            relative_path = blob.name[len(gcp_prefix):].lstrip('/')
            local_file = local_path / relative_path
            
            # Create parent directories
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Check if file exists and is same size
                if local_file.exists():
                    local_size = local_file.stat().st_size
                    if blob.size == local_size:
                        print(f"⏭️  Skipped (unchanged): {relative_path}")
                        stats["skipped"] += 1
                        continue
                
                # Download
                blob.download_to_filename(str(local_file))
                file_size_mb = blob.size / (1024 * 1024)
                stats["total_size_mb"] += file_size_mb
                stats["downloaded"] += 1
                
                print(f"✓ Downloaded: {relative_path} ({file_size_mb:.2f} MB)")
                
            except Exception as e:
                print(f"✗ Failed: {relative_path} - {e}")
                stats["failed"] += 1
        
        print(f"\n{'='*70}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*70}")
        print(f"Downloaded: {stats['downloaded']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        print(f"{'='*70}\n")
        
        return stats

    def list_documents(self, gcp_prefix: str = "processed_documents") -> List[str]:
        """
        List all documents in GCP bucket.
        
        Args:
            gcp_prefix: Prefix in GCP bucket
            
        Returns:
            List of document names
        """
        blobs = self.bucket.list_blobs(prefix=gcp_prefix, delimiter='/')
        
        # Get unique document folders
        documents = set()
        for blob in blobs:
            parts = blob.name[len(gcp_prefix):].strip('/').split('/')
            if parts and parts[0]:
                documents.add(parts[0])
        
        return sorted(list(documents))


def main():
    """Main function for GCP sync."""
    import sys
    
    print("\n" + "="*70)
    print("GCP BUCKET SYNC")
    print("="*70)
    
    # Check for GCP credentials
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        print("\n⚠️  GOOGLE_APPLICATION_CREDENTIALS not set!")
        print("Please set the path to your GCP service account key JSON file:")
        print("  export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
        print("\nOr add to .env file:")
        print("  GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        return
    
    # Initialize sync
    try:
        sync = GCPBucketSync()
    except ValueError as e:
        print(f"\n✗ {e}")
        print("\nAdd to .env file:")
        print("  GCP_BUCKET_NAME=your-bucket-name")
        return
    
    # Get action
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
    else:
        print("\nAvailable actions:")
        print("  upload   - Upload processed_documents to GCP")
        print("  download - Download from GCP to local")
        print("  list     - List documents in GCP bucket")
        action = input("\nSelect action: ").strip().lower()
    
    if action == "upload":
        # Upload processed_documents folder
        local_folder = "processed_documents"
        if not Path(local_folder).exists():
            print(f"\n✗ Folder not found: {local_folder}")
            return
        
        stats = sync.sync_folder(local_folder)
        
        print(f"\n✓ Sync complete!")
        print(f"View in GCP Console:")
        print(f"  https://console.cloud.google.com/storage/browser/{sync.bucket_name}/processed_documents")
    
    elif action == "download":
        # Download from GCP
        gcp_prefix = input("GCP prefix (default: processed_documents): ").strip() or "processed_documents"
        local_folder = input("Local folder (default: processed_documents_downloaded): ").strip() or "processed_documents_downloaded"
        
        stats = sync.download_folder(gcp_prefix, local_folder)
        
        print(f"\n✓ Download complete!")
        print(f"Files saved to: {local_folder}")
    
    elif action == "list":
        # List documents
        documents = sync.list_documents()
        
        print(f"\nDocuments in bucket ({len(documents)}):")
        for doc in documents:
            print(f"  - {doc}")
    
    else:
        print(f"\n✗ Unknown action: {action}")
        print("Valid actions: upload, download, list")


if __name__ == "__main__":
    main()
