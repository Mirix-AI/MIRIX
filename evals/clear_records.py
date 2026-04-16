#!/usr/bin/env python3
"""
Script to remove 'records' key from JSON files in a specified folder.
If metrics.json exists, only removes records marked as WRONG.
If metrics.json doesn't exist, removes all records.

Usage:
    python clear_records.py <folder_path>
    
Example:
    python clear_records.py results/0201a
"""

import json
import sys
from pathlib import Path
from typing import Optional, Set, Tuple


def load_wrong_records(metrics_path: Path) -> Set[Tuple[str, int]]:
    """
    Load metrics.json and extract sample_id and question_index for WRONG records.
    
    Args:
        metrics_path: Path to metrics.json file
        
    Returns:
        Set of (sample_id, question_index) tuples for wrong records
    """
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        wrong_records = set()
        llm_judge_results = metrics.get('llm_judge_results', [])
        
        for result in llm_judge_results:
            if result.get('label') == 'WRONG' or result.get('score') == 0:
                sample_id = result.get('sample_id')
                question_index = result.get('question_index')
                if sample_id and question_index:
                    wrong_records.add((sample_id, question_index))
        
        return wrong_records
    except Exception as e:
        print(f"  ✗ Error loading metrics.json: {e}")
        return set()


def clear_records_from_file(file_path: Path, wrong_records: Optional[Set[Tuple[str, int]]] = None) -> bool:
    """
    Remove 'records' key from a JSON file if it exists.
    If wrong_records is provided, only removes those specific records.
    If wrong_records is None, removes all records.
    
    Args:
        file_path: Path to the JSON file
        wrong_records: Optional set of (sample_id, question_index) tuples to remove
        
    Returns:
        True if the file was modified, False otherwise
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if 'records' key exists
        if 'records' in data:
            sample_id = data.get('sample_id')
            
            if wrong_records is None:
                # Remove all records (original behavior)
                print(f"  Found 'records' key in {file_path.name}")
                del data['records']
                print(f"  ✓ Removed all records from {file_path.name}")
                modified = True
            else:
                # Remove only wrong records
                original_count = len(data['records'])
                records_to_keep = {}
                removed_count = 0
                
                for key, record in data['records'].items():
                    question_index = record.get('question_index')
                    if (sample_id, question_index) not in wrong_records:
                        records_to_keep[key] = record
                    else:
                        removed_count += 1
                
                if removed_count > 0:
                    data['records'] = records_to_keep
                    print(f"  ✓ Removed {removed_count}/{original_count} wrong records from {file_path.name}")
                    modified = True
                else:
                    print(f"  No wrong records found in {file_path.name}")
                    modified = False
            
            # Write back to file if modified
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            return modified
        else:
            print(f"  No 'records' key in {file_path.name}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"  ✗ Error decoding JSON in {file_path.name}: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error processing {file_path.name}: {e}")
        return False


def clear_records_from_folder(folder_path: str) -> None:
    """
    Process all JSON files in a folder and remove 'records' key.
    If metrics.json exists, only removes records marked as WRONG.
    If metrics.json doesn't exist, removes all records.
    
    Args:
        folder_path: Path to the folder containing JSON files
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)
    
    # Check for metrics.json
    metrics_path = folder / 'metrics.json'
    wrong_records = None
    
    if metrics_path.exists():
        print(f"Found metrics.json - will only remove WRONG records\n")
        wrong_records = load_wrong_records(metrics_path)
        if wrong_records:
            print(f"Identified {len(wrong_records)} wrong record(s) to remove\n")
        else:
            print("No wrong records found in metrics.json\n")
    else:
        print(f"No metrics.json found - will remove all records\n")
    
    # Find all JSON files in the folder (excluding metrics.json)
    json_files = [f for f in folder.glob('*.json') if f.name != 'metrics.json']
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return
    
    print(f"Processing {len(json_files)} JSON file(s) in '{folder_path}'...\n")
    
    modified_count = 0
    for json_file in sorted(json_files):
        if clear_records_from_file(json_file, wrong_records):
            modified_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files processed: {len(json_files)}")
    print(f"  Files modified: {modified_count}")
    print(f"  Files unchanged: {len(json_files) - modified_count}")
    if wrong_records is not None:
        print(f"  Wrong records removed: {len(wrong_records)}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python clear_records.py <folder_path>")
        print("\nExample:")
        print("  python clear_records.py results/0201a")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    clear_records_from_folder(folder_path)


if __name__ == "__main__":
    main()
