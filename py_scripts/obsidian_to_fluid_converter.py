import os
import re
import argparse
import shutil
from typing import List

# 定义 Obsidian Callout 类型到 Fluid Note 类型的映射
# 您可以根据自己的需求添加或修改这个映射
# Fluid 支持的类型: default, primary, success, info, warning, danger
OBSIDIAN_TO_FLUID_MAP = {
    'note': 'default',
    'abstract': 'default',
    'summary': 'default',
    'tldr': 'default',
    'info': 'info',
    'todo': 'info',
    'tip': 'success',
    'hint': 'success',
    'success': 'success',
    'check': 'success',
    'done': 'success',
    'question': 'warning',
    'help': 'warning',
    'faq': 'warning',
    'warning': 'warning',
    'caution': 'warning',
    'attention': 'warning',
    'failure': 'danger',
    'fail': 'danger',
    'missing': 'danger',
    'danger': 'danger',
    'error': 'danger',
    'bug': 'danger',
    'example': 'primary',
    'quote': 'default',
    'cite': 'default'
}

def convert_content(content: str) -> str:
    """
    转换单个文件的内容，将 Obsidian Callouts 转换为 Fluid Note 标签。
    (已修复处理内部空行的问题)
    """
    lines = content.splitlines()
    new_lines = []
    in_callout = False
    
    # 正则表达式，用于匹配 Callout 的起始行
    # 例如：> [!note] My Title
    callout_pattern = re.compile(r"^\s*>\s*\[!(?P<type>\w+)\]\s*(?P<title>.*)", re.IGNORECASE)

    for line in lines:
        if not in_callout:
            match = callout_pattern.match(line)
            if match:
                # 匹配到了 Callout 的开始
                in_callout = True
                callout_type = match.group('type').lower()
                title = match.group('title').strip()

                # 从映射中获取 Fluid 的类型，如果找不到则默认为 'default'
                fluid_type = OBSIDIAN_TO_FLUID_MAP.get(callout_type, 'default')
                
                # 添加 Fluid Note 的起始标签
                new_lines.append(f"{{% note {fluid_type} '{title}' %}}")
            else:
                # 普通行，直接添加
                new_lines.append(line)
        else:
            # 已经在 Callout 内部
            is_blockquote = line.strip().startswith('>')
            is_blank_line = not line.strip()

            if is_blockquote:
                # 移除行首的 '>' 和一个可选的空格
                content_line = re.sub(r"^\s*>\s?", "", line)
                new_lines.append(content_line)
            elif is_blank_line:
                # 如果是空行，则在笔记块内部保留一个空行
                new_lines.append("")
            else:
                # 如果是带内容的非引用行，则 Callout 结束
                in_callout = False
                # 添加 Fluid Note 的结束标签
                new_lines.append("{% endnote %}")
                # 将当前行（非 Callout 的第一行）也添加进去
                new_lines.append(line)

    # 如果文件以 Callout 结尾，确保闭合标签
    if in_callout:
        new_lines.append("{% endnote %}")
        
    return "\n".join(new_lines)


def process_file(file_path: str, create_backup: bool):
    """
    处理单个 Markdown 文件。
    """
    try:
        print(f"正在处理文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        converted_content = convert_content(original_content)

        if original_content != converted_content:
            if create_backup:
                backup_path = file_path + '.bak'
                shutil.copy2(file_path, backup_path)
                print(f"  -> 已创建备份: {backup_path}")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(converted_content)
            print(f"  -> 文件已转换并保存。")
        else:
            print(f"  -> 文件无需转换。")
            
    except Exception as e:
        print(f"  -> 处理文件时发生错误: {e}")


def main():
    """
    主函数，用于解析命令行参数并启动转换过程。
    """
    parser = argparse.ArgumentParser(
        description="将 Markdown 文件中的 Obsidian Callouts 转换为 Hexo Fluid 主题的 Note 标签。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "path",
        help="要处理的文件或目录的路径。"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份文件。警告：此操作会直接覆盖原文件！"
    )
    
    args = parser.parse_args()

    target_path = args.path
    create_backup = not args.no_backup

    if not os.path.exists(target_path):
        print(f"错误: 路径 '{target_path}' 不存在。")
        return

    if os.path.isfile(target_path):
        if target_path.lower().endswith('.md'):
            process_file(target_path, create_backup)
        else:
            print(f"跳过非 Markdown 文件: {target_path}")
    elif os.path.isdir(target_path):
        print(f"开始扫描目录: {target_path}")
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith('.md'):
                    file_path = os.path.join(root, file)
                    process_file(file_path, create_backup)
    
    print("\n所有操作已完成！")


if __name__ == "__main__":
    main()


