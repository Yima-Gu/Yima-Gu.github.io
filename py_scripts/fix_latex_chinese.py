import os
import re
import argparse
from pathlib import Path

def replace_chinese_in_latex_match(match):
    """
    Regex replacer callback.
    Receives a match object containing a LaTeX formula (including delimiters).
    Replaces Chinese characters inside with \text{...} while preserving existing \text{...} blocks.
    """
    latex_content = match.group(0)
    
    placeholders = []
    
    # 1. Protect existing \text{...} blocks to avoid double wrapping (e.g. \text{求解} becoming \text{求\text{解}})
    # Pattern explanation:
    # \\text\{        : Match literal \text{
    # (               : Start capturing group 1 (the content)
    #   (?:           : Start non-capturing group (for alternatives)
    #     \\.         : Match any escaped character (e.g. \{, \})
    #     |           : OR
    #     [^{}]       : Match any character that is NOT { or }
    #   )*            : Repeat 0 or more times
    # )               : End capturing group
    # \}              : Match literal }
    # This handles simple nesting like \{ \} but not full recursive nesting { { } }. 
    # For standard Chinese math labels, this is usually sufficient.
    existing_text_pattern = re.compile(r'\\text\{((?:\\.|[^{}])*)\}')
    
    def protect_callback(m):
        # Save the original match (e.g. \text{求解})
        placeholders.append(m.group(0))
        # Return a unique placeholder
        return f"__LATEX_TEXT_PROTECTED_{len(placeholders)-1}__"
    
    # Replace valid \text{...} blocks with placeholders
    temp_content = existing_text_pattern.sub(protect_callback, latex_content)
    
    # 2. Wrap remaining Chinese characters in the content (which now lacks the protected parts)
    # Regex to find Chinese characters: \u4e00-\u9fa5
    chinese_char_pattern = re.compile(r'([\u4e00-\u9fa5]+)')
    
    def wrap_in_text(m):
        return r'\text{' + m.group(1) + '}'
    
    # Perform the replacement
    fixed_content = chinese_char_pattern.sub(wrap_in_text, temp_content)
    
    # 3. Restore the protected \text{...} blocks
    for i, original_text in enumerate(placeholders):
        fixed_content = fixed_content.replace(f"__LATEX_TEXT_PROTECTED_{i}__", original_text)
    
    return fixed_content

def process_file(file_path: Path, create_backup: bool = True):
    try:
        # Read file content
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Regex to capture LaTeX formulas
    # 1. Block Math: $$ ... $$ (Checking for unescaped $$)
    # 2. Inline Math: $ ... $ (Checking for unescaped $)
    # Note: We use non-capturing groups (?:) for inner content to keep logic simple
    
    # Pattern breakdown:
    # (?<!\\)\$\$       : Match $$ not preceded by \
    # [\s\S]*?          : Match any character (including newlines) non-greedy
    # (?<!\\)\$\$       : Match $$ not preceded by \
    # |                 : OR
    # (?<!\\)\$         : Match $ not preceded by \
    # (?:\\.|[^$\n])*?  : Match escaped chars OR non-$ non-newline chars (non-greedy)
    # (?<!\\)\$         : Match $ not preceded by \
    
    math_pattern = re.compile(r'(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$|(?<!\\)\$(?:\\.|[^$\n])*?(?<!\\)\$')
    
    # Substitute using the callback function
    new_content = math_pattern.sub(replace_chinese_in_latex_match, content)

    # Check if changes were made
    if new_content != content:
        print(f"Fixing: {file_path}")
        
        if create_backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            try:
                backup_path.write_text(content, encoding='utf-8')
                print(f"  -> Backup created: {backup_path.name}")
            except Exception as e:
                print(f"  -> Failed to create backup: {e}")
                return # Stop if backup fails specifically? Or continue? Usually safety first.
        
        try:
            file_path.write_text(new_content, encoding='utf-8')
            print(f"  -> Saved changes.")
        except Exception as e:
            print(f"  -> Failed to save file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan Markdown files and wrap Chinese characters in LaTeX formulas with \\text{}."
    )
    parser.add_argument(
        "root_dir", 
        nargs='?', 
        default=".", 
        help="Root directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true", 
        help="Do not create .bak backup files."
    )
    
    args = parser.parse_args()
    
    root_path = Path(args.root_dir).resolve()
    
    if not root_path.exists():
        print(f"Error: Directory '{root_path}' does not exist.")
        return

    print(f"Scanning directory: {root_path}")
    print("Looking for Chinese characters inside $...$ and $$...$$ ...")

    count = 0
    # Recursive glob patterns for .md files
    for file_path in root_path.rglob("*.md"):
        process_file(file_path, create_backup=not args.no_backup)
        count += 1
        
    print(f"Done. Scanned {count} files.")

if __name__ == "__main__":
    main()
