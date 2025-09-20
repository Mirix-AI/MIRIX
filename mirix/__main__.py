#!/usr/bin/env python3
"""
Main entry point for Mirix module when called with python -m mirix
"""

import sys
import argparse

def main():
    """Main entry point for module-level commands."""
    parser = argparse.ArgumentParser(description='Mirix CLI Tools')
    parser.add_argument('subcommand', choices=['db-maintenance'], 
                       help='Subcommand to run')
    parser.add_argument('args', nargs='*', help='Arguments for the subcommand')
    
    args, unknown = parser.parse_known_args()
    
    if args.subcommand == 'db-maintenance':
        from mirix.db_maintenance import main as db_main
        # Reconstruct sys.argv for the subcommand
        sys.argv = ['db_maintenance'] + args.args + unknown
        db_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
