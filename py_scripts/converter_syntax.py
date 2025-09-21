# converter_syntax.py (已修正版本)
import re
import sys
import os
import shutil
import argparse

def convert_syntax(content):
    """
    转换 Markdown 文件中的语法，只负责文本替换。
    1. ==高亮== -> <mark>高亮</mark>
    2. ![|宽度](路径) -> <img src="文件名" alt="" width="宽度">
    3. ![[文件名|宽度]] -> <img src="文件名" alt="" width="宽度"> 或 ![[文件名]] -> ![文件名](文件名)
    4. ![alt](路径) -> ![alt](文件名)
    """
    
    # --- 1. 转换 ==高亮== 语法 ---
    content = re.sub(r'==(.+?)==', r'<mark>\1</mark>', content)

    # --- 2. 转换 Obsidian 带尺寸的图片语法 ![|width](path) ---
    def replace_obsidian_image(match):
        width = match.group(1)
        path = match.group(2)
        filename = os.path.basename(path)
        return f'<img src="{filename}" alt="" width="{width}">'
    content = re.sub(r'!\[\|(\d+)\]\((.*?)\)', replace_obsidian_image, content)

    # --- 3. 新增：转换 Obsidian Wiki 图片语法 ![[file|width]] ---
    def replace_wiki_image(match):
        path = match.group(1).strip() # 获取文件名
        width = match.group(2)      # 获取可选的宽度
        filename = os.path.basename(path)
        
        if width:
            # 如果有宽度，转换为带尺寸的 <img> 标签
            return f'<img src="{filename}" alt="" width="{width}">'
        else:
            # 如果没有宽度，转换为标准的 Markdown 图片格式
            alt_text = os.path.splitext(filename)[0] # 使用文件名（不含扩展名）作为 alt 文本
            return f'![{alt_text}]({filename})'
            
    # 正则表达式匹配 ![[文件名]] 或 ![[文件名|宽度]]
    wiki_image_regex = re.compile(r'!\[\[([^|\]\n]+)(?:\|(\d+))?\]\]')
    content = wiki_image_regex.sub(replace_wiki_image, content)

    # --- 4. 转换标准 Markdown 图片语法 ![alt](path) ---
    def replace_standard_image(match):
        alt_text = match.group(1)
        path = match.group(2)
        filename = os.path.basename(path)
        return f'![{alt_text}]({filename})'
    # 这个正则表达式必须在最后，因为它最通用
    standard_image_regex = re.compile(r'!\[(.*?)\]\((.*?)\)')
    content = standard_image_regex.sub(replace_standard_image, content)

    return content

def process_single_file(filepath, create_backup):
    """
    处理单个 markdown 文件，包括移动图片和转换语法。
    """
    print(f"正在检查: {filepath}")
    try:
        dir_path = os.path.dirname(filepath)
        post_filename = os.path.basename(filepath)
        post_name = os.path.splitext(post_filename)[0]
        
        source_attachments_dir = os.path.join(dir_path, 'attachments')
        dest_asset_dir = os.path.join(dir_path, post_name) # 与 md 文件同名的资源文件夹

        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        fm_match = re.match(r'---\s*?\n(.*?)\n---\s*?\n', original_content, re.DOTALL)
        if fm_match and 'syntax_converted: true' in fm_match.group(1):
            print("  -> 跳过: 文件已被标记为已转换。")
            return

        images_to_move = set()
        wiki_links = re.findall(r'!\[\[([^|\]\n]+)(?:\|(?:\d+))?\]\]', original_content)
        for link in wiki_links:
            images_to_move.add(os.path.basename(link.strip()))
            
        md_links = re.findall(r'!\[.*?\]\((attachments/[^)]+)\)', original_content)
        for link in md_links:
            images_to_move.add(os.path.basename(link.strip()))

        if images_to_move:
            if not os.path.isdir(source_attachments_dir):
                print(f"  -> 警告: 找不到源图片目录 {source_attachments_dir}，跳过图片移动。")
            else:
                print(f"  -> 检测到 {len(images_to_move)} 张图片需要处理，目标文件夹: {dest_asset_dir}")
                os.makedirs(dest_asset_dir, exist_ok=True) 

                for image_file in images_to_move:
                    source_image_path = os.path.join(source_attachments_dir, image_file)
                    dest_image_path = os.path.join(dest_asset_dir, image_file)
                    
                    if os.path.exists(source_image_path):
                        print(f"    -> 正在移动: {image_file}")
                        shutil.move(source_image_path, dest_image_path)
                    else:
                        print(f"    -> 警告: 在 attachments 中未找到图片 {image_file}")
                
                if not os.listdir(source_attachments_dir):
                    print(f"  -> 源图片目录 {source_attachments_dir} 已清空，正在删除...")
                    os.rmdir(source_attachments_dir)
        
        converted_content = convert_syntax(original_content)

        if original_content != converted_content:
            if create_backup:
                backup_path = filepath + ".bak"
                if not os.path.exists(backup_path):
                    shutil.copy2(filepath, backup_path)
                    print(f"  -> 已创建备份: {backup_path}")
            
            if fm_match:
                front_matter_content = fm_match.group(1)
                if 'syntax_converted: true' not in front_matter_content:
                    new_front_matter = front_matter_content.strip() + "\nsyntax_converted: true\n"
                    final_content = original_content.replace(front_matter_content, new_front_matter, 1)
                    final_content = final_content.replace(original_content[fm_match.end():], converted_content[fm_match.end():], 1)
                else:
                    final_content = converted_content
            else:
                new_front_matter = "syntax_converted: true\n"
                final_content = f"---\n{new_front_matter}---\n\n{converted_content}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(final_content)
            print("  -> 图片已移动，语法已转换，并已标记文件。")
        else:
            print("  -> 无需转换任何语法。")

    except Exception as e:
        print(f"  -> 处理文件 {filepath} 时出错: {e}", file=sys.stderr)

# +++ 添加这个缺失的函数 +++
def process_path(path, create_backup):
    """
    递归处理给定路径（文件或目录）中的文件。
    """
    if os.path.isfile(path):
        if path.endswith(".md"):
            process_single_file(path, create_backup)
        else:
            print(f"跳过非 markdown 文件: {path}")
    elif os.path.isdir(path):
        print(f"正在处理目录: {path}")
        for root, _, files in os.walk(path):
            # 过滤掉不想处理的目录，例如 .git, node_modules 等
            if '.git' in root or 'node_modules' in root:
                continue
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    process_single_file(file_path, create_backup)
    else:
        print(f"错误: 路径不存在: {path}", file=sys.stderr)
# ++++++++++++++++++++++++


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
转换 Markdown 文件以适配 Hexo 的 'post_asset_folder' 模式。
功能:
1. 移动图片: 将 'attachments' 目录下的图片移动到与文章同名的资源文件夹中。
2. 转换链接: 处理标准、Obsidian尺寸和Wiki图片链接，移除路径。
3. 转换语法: 将 ==高亮== 转换为 <mark> 标签。
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("paths", metavar="PATH", type=str, nargs='+', help="一个或多个要处理的 markdown 文件或目录的路径。")
    parser.add_argument("--no-backup", action="store_false", dest="create_backup", help="禁用创建 .bak 备份文件。")
    
    args = parser.parse_args()
    
    print("重要提示: 此脚本会物理移动您的图片文件，并修改您的 markdown 源文件。")
    print("它会将 'attachments' 目录下的图片移动到与 markdown 文件同名的文件夹中。\n")

    for path in args.paths:
        process_path(path, args.create_backup)
        
    print("\n转换过程完成。")