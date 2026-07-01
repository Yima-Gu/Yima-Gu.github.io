# Yima Gu's Personal Website

个人技术笔记站，基于 Hexo + Fluid 主题构建。

[![Website](https://img.shields.io/badge/website-yima--gu.github.io-blue)](https://yima-gu.github.io)
[![Hexo](https://img.shields.io/badge/hexo-8.x-blue)](https://hexo.io)

**网站地址**: [https://yima-gu.github.io](https://yima-gu.github.io)

## 内容方向

- 计算机网络课程笔记
- CSAPP 读书笔记
- 算法导论学习笔记
- 深度学习笔记
- 形式语言与自动机

## 技术栈

| 技术 | 用途 |
|------|------|
| [Hexo](https://hexo.io/) | 静态网站生成器 |
| [Fluid Theme](https://github.com/fluid-dev/hexo-theme-fluid) | 主题框架 |
| [KaTeX](https://katex.org/) | 数学公式渲染 |
| [Giscus](https://giscus.app/) | 评论系统 |
| [GitHub Actions](https://github.com/features/actions) | 自动部署到 GitHub Pages |

## 本地开发

### 环境要求

- Node.js >= 18
- npm >= 8

### 快速开始

```bash
git clone https://github.com/Yima-Gu/Yima-Gu.github.io.git
cd Yima-Gu.github.io
npm install
npm run server
```

访问 [http://localhost:4000](http://localhost:4000) 预览。

### 常用命令

```bash
npm run clean    # 清理缓存
npm run build    # 生成静态文件
npm run server   # 本地预览
```

推送至 `main` 分支后，GitHub Actions 会自动构建并部署到 `gh-pages` 分支。

### 发布新文章

```bash
# 从 Obsidian 导出后，先预处理 Markdown（图片迁移、语法转换）
./publish.sh source/_posts/your-post.md

# 或仅本地预览
npx hexo new post "文章标题"
npm run server
```

## 项目结构

```text
├── source/
│   ├── _posts/        # 博客文章
│   ├── about/         # 关于页面
│   ├── series/        # 系列笔记索引
│   └── img/           # 图片资源
├── _config.yml        # Hexo 主配置
├── _config.fluid.yml  # Fluid 主题配置
├── py_scripts/        # Markdown 预处理脚本
└── .github/workflows/ # CI 部署
```

## 功能特性

- 站内搜索（hexo-generator-search）
- RSS 订阅（`/atom.xml`）
- Sitemap（`/sitemap.xml`）
- 数学公式（KaTeX）
- Mermaid 图表
- Giscus 评论
- Google Analytics 统计

## 联系

- GitHub: [Yima-Gu](https://github.com/Yima-Gu)
- Email: yima.gu.23@gmail.com

## 许可证

MIT License
