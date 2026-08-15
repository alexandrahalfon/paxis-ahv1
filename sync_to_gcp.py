#!/usr/bin/env python3
"""
Sync Processed Documents to GCP Bucket
Modified to skip existence checks (only needs write permission)
"""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


class GCPBucketSync:
    """Sync processed documents to GCP bucket."""

    def __init__(self, bucket_name: Optional[str] = None):
        """Initialize GCP sync."""
        from google.cloud import storage
        
        self.bucket_name = bucket_name or os.getenv('GCP_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("GCP_BUCKET_NAME not set. Please set in .env file or pass as argument.")
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        
        print(f"✓ Connected to GCP bucket: {self.bucket_name}")

    def sync_document(self, doc_name: str, category: str = None, local_base: str = "processed_documents",
                     gcp_prefix: str = "processed_documents") -> dict:
        """
        Sync a single document folder to GCP.
        MODIFIED: Skips existence check, always uploads (write-only permission)
        """
        local_folder = Path(local_base) / category / doc_name if category else Path(local_base) / doc_name
        
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
                gcp_path = f"{gcp_prefix}/{category}/{doc_name}/{relative_path}".replace("\\", "/") if category else f"{gcp_prefix}/{doc_name}/{relative_path}".replace("\\", "/")
                
                try:
                    blob = self.bucket.blob(gcp_path)
                    
                    # REMOVED: blob.exists() check (requires read permission)
                    # Just upload directly
                    blob.upload_from_filename(str(local_file))
                    file_size_mb = local_file.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += file_size_mb
                    stats["uploaded"] += 1
                    
                    print(f"  ✓ {relative_path}")
                    
                except Exception as e:
                    print(f"  ✗ {relative_path} - {e}")
                    stats["failed"] += 1
        
        print(f"  Uploaded: {stats['uploaded']}, Failed: {stats['failed']}")
        
        return stats


def main():
    """Main function for GCP sync."""
    import sys
    
    print("\n" + "="*70)
    print("GCP BUCKET SYNC")
    print("="*70)
    
    try:
        sync = GCPBucketSync()
    except ValueError as e:
        print(f"\n✗ {e}")
        return
    
    if len(sys.argv) < 2:
        print("\nUsage: python sync_to_gcp.py <document_name>")
        print("Example: python sync_to_gcp.py doi_10.1056_nejmoa1213755")
        return
    
    doc_name = sys.argv[1]
    stats = sync.sync_document(doc_name)
    
    print(f"\n✓ Sync complete!")
    print(f"View in GCP Console:")
    print(f"  https://console.cloud.google.com/storage/browser/{sync.bucket_name}/processed_documents/{doc_name}")


if __name__ == "__main__":
    main()
