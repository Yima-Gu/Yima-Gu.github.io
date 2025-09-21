# Yima Gu's Personal Website

🌐 我的个人技术博客 - 基于 Hexo + Fluid 主题构建的现代化个人网站

[![Website](https://img.shields.io/badge/website-yima--gu.github.io-blue)](https://yima-gu.github.io)
[![Hexo](https://img.shields.io/badge/hexo-7.3.0-blue)](https://hexo.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 网站简介

这是我的个人技术博客，专注于分享技术学习心得、项目经验和个人思考。网站采用现代化的设计理念，提供优质的阅读体验。

**网站地址**: [https://yima-gu.github.io](https://yima-gu.github.io)

### ✨ 主要特色

- 🎨 **现代化设计** - 采用 Fluid 主题，响应式布局，支持暗黑模式
- 📝 **优质内容** - 技术分享、学习笔记、项目经验
- 📐 **数学公式支持** - 基于 MathJax 的完美 LaTeX 公式渲染
- 🚀 **快速加载** - 优化的静态网站，CDN 加速
- 🔍 **搜索友好** - SEO 优化，支持本地搜索
- 📱 **移动端适配** - 完美支持各种设备

## 🛠️ 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| [Hexo](https://hexo.io/) | 7.3.0 | 静态网站生成器 |
| [Fluid Theme](https://github.com/fluid-dev/hexo-theme-fluid) | 1.9.8 | 主题框架 |
| [Pandoc](https://pandoc.org/) | Latest | Markdown 渲染引擎 |
| [MathJax](https://www.mathjax.org/) | 3.x | 数学公式渲染 |
| [GitHub Pages](https://pages.github.com/) | - | 网站托管 |

## 🚀 本地开发

### 环境要求

- Node.js >= 16.0.0
- npm >= 8.0.0
- Git

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/Yima-Gu/personal_blog.git
cd personal_blog

# 安装依赖
npm install

# 本地预览
npm run server
```

访问 [http://localhost:4000](http://localhost:4000) 预览网站

### 常用命令

```bash
# 清理缓存
npm run clean

# 生成静态文件
npm run build

# 部署到 GitHub Pages
npm run deploy

# 创建新文章
npx hexo new post "文章标题"
```

## 📝 内容创作

### 文章结构

```yaml
---
title: 文章标题
date: 2025-07-05
categories: [技术分享]
tags: [JavaScript, React]
description: 文章描述
---

文章内容...
```

### 支持功能

- **Markdown** - 标准 Markdown 语法
- **代码高亮** - 支持多种编程语言
- **数学公式** - LaTeX 语法支持
- **图片优化** - 自动压缩和懒加载
- **内链支持** - 文章间相互引用

## 📁 项目结构

```text
personal_blog/
├── source/                 # 源文件目录
│   ├── _posts/            # 博客文章
│   ├── about/             # 关于页面
│   └── images/            # 图片资源
├── themes/                # 主题文件
├── public/                # 生成的静态文件
├── _config.yml           # Hexo 主配置
├── _config.fluid.yml     # Fluid 主题配置
├── package.json          # 项目依赖
└── README.md             # 项目说明
```

## 🎨 自定义配置

### 网站信息

在 `_config.yml` 中修改基本信息：

```yaml
title: Yima Gu's Blog
author: Yima Gu
description: 个人技术博客
url: https://yima-gu.github.io
```

### 主题配置

在 `_config.fluid.yml` 中自定义主题：

```yaml
navbar:
  blog_title: "Yima Gu"
  
index:
  slogan:
    enable: true
    text: "代码改变世界"
```

## 📈 网站统计

- 📊 集成 Google Analytics
- 🔍 支持本地搜索
- 💬 Gitalk 评论系统
- 📱 移动端优化

## 🤝 贡献与反馈

如果您发现任何问题或有改进建议，欢迎：

- 提交 [Issue](https://github.com/Yima-Gu/personal_blog/issues)
- 发起 [Pull Request](https://github.com/Yima-Gu/personal_blog/pulls)
- 通过邮件联系: [your.email@example.com](mailto:your.email@example.com)

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)

## 🙏 致谢

感谢以下开源项目的支持：

- [Hexo](https://hexo.io/) - 优秀的静态网站生成器
- [Fluid](https://github.com/fluid-dev/hexo-theme-fluid) - 美观的 Hexo 主题
- [GitHub Pages](https://pages.github.com/) - 免费的网站托管服务

---

⭐ 如果这个项目对您有帮助，请给个 Star 支持一下！

💼 **Let's connect**: 欢迎关注我的技术分享和项目更新
