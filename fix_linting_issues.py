#!/usr/bin/env python3
"""
Script to automatically fix common linting issues in the codebase.
"""

import os
import re
import sys
from pathlib import Path


def fix_f_string_placeholders(content: str) -> str:
    """Fix f-strings that are missing placeholders by converting them to regular strings."""
    # Find f-strings without any {} placeholders
    pattern = r'f"([^"{}]*)"'
    
    def replace_f_string(match):
        inner_content = match.group(1)
        # Only replace if there are no placeholders
        if '{' not in inner_content and '}' not in inner_content:
            return f'"{inner_content}"'
        return match.group(0)
    
    content = re.sub(pattern, replace_f_string, content)
    
    # Also handle single quotes
    pattern = r"f'([^'{}]*)'"
    content = re.sub(pattern, replace_f_string, content)
    
    return content


def remove_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from all lines."""
    lines = content.split('\n')
    fixed_lines = [line.rstrip() for line in lines]
    return '\n'.join(fixed_lines)


def fix_bare_excepts(content: str) -> str:
    """Fix bare except clauses by adding Exception."""
    # Replace bare except: with except Exception:
    content = re.sub(r'\bexcept\s*:\s*$', 'except Exception:', content, flags=re.MULTILINE)
    return content


def fix_line_length_simple(content: str) -> str:
    """Fix some simple line length issues."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) > 100:
            # Try to break long import lines
            if line.strip().startswith('from ') and ' import ' in line:
                # Split long import lines
                if ', ' in line and len(line) > 100:
                    parts = line.split(' import ')
                    if len(parts) == 2:
                        from_part = parts[0]
                        imports = [imp.strip() for imp in parts[1].split(',')]
                        
                        # If it's still too long, break it up
                        if len(line) > 120:
                            fixed_line = from_part + ' import (\n'
                            for imp in imports:
                                fixed_line += f'    {imp},\n'
                            fixed_line += ')'
                            fixed_lines.append(fixed_line)
                            continue
            
            # For other long lines, just keep them as is for manual fixing
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_file(file_path: Path) -> bool:
    """Fix a single Python file and return True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        content = original_content
        
        # Apply fixes
        content = fix_f_string_placeholders(content)
        content = remove_trailing_whitespace(content)
        content = fix_bare_excepts(content)
        content = fix_line_length_simple(content)
        
        # Check if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix linting issues."""
    root_dir = Path('.')
    
    # Find all Python files in the project
    python_files = []
    for pattern in ['fm_llm_solver/**/*.py', 'tests/**/*.py']:
        python_files.extend(root_dir.glob(pattern))
    
    # Remove duplicates and filter out __pycache__ directories
    python_files = [f for f in set(python_files) if '__pycache__' not in str(f)]
    
    print(f"Found {len(python_files)} Python files to check")
    
    fixed_count = 0
    for file_path in sorted(python_files):
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")
    
    if fixed_count > 0:
        print("\nRecommended next steps:")
        print("1. Run Black formatter: black fm_llm_solver/ tests/")
        print("2. Remove unused imports manually or with a tool like autoflake")
        print("3. Check remaining flake8 issues: flake8 fm_llm_solver/ tests/ --max-line-length=100 --extend-ignore=E203,W503")


if __name__ == '__main__':
    main() 