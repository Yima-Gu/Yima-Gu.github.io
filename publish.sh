#!/bin/bash

# publish.sh - 支持对单个文件或整个目录进行预处理的全功能脚本

# 若有命令失败则立即退出
set -e

# --- 配置 ---
# 定义 Python 脚本的路径
PYTHON_PREPROCESSOR="py_scripts/converter_syntax.py"
# 定义默认要处理的源目录
DEFAULT_SOURCE_DIR="source/"

# 将要处理的目标路径初始化为空
TARGET_PATH=""

# --- ANSI 颜色代码 ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- 参数解析 ---
# 检查是否存在第一个参数，并且该参数不是一个标志 (不以'-'开头)
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    # 如果是，则将其视为目标文件路径
    TARGET_PATH="$1"
    echo -e "${CYAN}检测到指定文件，预处理将只针对: ${TARGET_PATH}${NC}"
else
    # 否则，使用默认的整个源目录
    TARGET_PATH="$DEFAULT_SOURCE_DIR"
    echo -e "${CYAN}未指定单个文件，预处理将针对整个目录: ${TARGET_PATH}${NC}"
fi

# --- 主要脚本逻辑 ---

# 步骤 1: 执行 Python 脚本进行预处理
echo -e "\n${YELLOW}>>> 步骤 1: 执行 Markdown 预处理 (移动图片、转换语法)...${NC}"
echo -e "    操作路径: ${CYAN}${TARGET_PATH}${NC}"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_PREPROCESSOR" ]; then
    echo -e "错误: Python 脚本 '$PYTHON_PREPROCESSOR' 未找到。"
    echo -e "请确认脚本路径是否正确。"
    exit 1
fi

# 调用 Python 脚本，并将目标路径作为参数传给它
python "$PYTHON_PREPROCESSOR" "$TARGET_PATH"

echo -e "${GREEN}--- 预处理完成。${NC}"


# --- 后续的 Hexo 命令 ---
echo -e "\n${YELLOW}重要提示: Hexo 的生成和部署命令是全局性的，将会根据整个 'source' 目录重新构建和发布您的网站。${NC}"

# 步骤 2: 清理旧文件
echo -e "\n${YELLOW}>>> 步骤 2: 清理旧的 Hexo 文件...${NC}"
hexo clean
echo -e "${GREEN}--- 清理完成。${NC}"

# 步骤 3: 生成新站点
echo -e "\n${YELLOW}>>> 步骤 3: 生成新的站点文件...${NC}"
hexo g
echo -e "${GREEN}--- 生成完成。${NC}"

# 步骤 4: 部署到 GitHub
echo -e "\n${YELLOW}>>> 步骤 4: 部署到 GitHub Pages...${NC}"
hexo d
echo -e "${GREEN}--- 部署完成。${NC}"

echo -e "\n${GREEN}✅ 所有步骤成功完成！你的博客已发布。${NC}"