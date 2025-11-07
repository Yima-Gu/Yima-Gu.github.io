# Yima Gu's Personal Website

🌐 我的个人技术博客 - 专注于分享技术学习心得、项目经验和个人思考
<br>
**[https://yima-gu.github.io](https://yima-gu.github.io)**

<p align="center">
  <a href="https://hexo.io/"><img src="https://img.shields.io/badge/Hexo-7.3.0-blue.svg?logo=hexo" alt="Hexo"></a>
  <a href="https://github.com/fluid-dev/hexo-theme-fluid"><img src="https://img.shields.io/badge/Theme-Fluid_1.9.8-0E83CD.svg" alt="Fluid Theme"></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/Node.js-%3E=16-green.svg?logo=node.js" alt="Node.js"></a>
  <a href="https://pages.github.com/"><img src="https://img.shields.io/badge/Host-GitHub_Pages-black.svg?logo=github" alt="GitHub Pages"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

## 博客内容

- 清华大学软件学院课程笔记
  - 形式语言与自动机
  - 算法分析与设计基础
  - 计算机网络
  - 计算机组成原理
  - 深度学习

## 🚀 快速上手 (本地开发)

1.  **克隆与安装**
    ```bash
    git clone https://github.com/Yima-Gu/Yima-Gu.github.io.git
    cd Yima-Gu.github.io
    npm install
    ```
2.  **本地预览**
    ```bash
    npm run server
    ```
    (访问 `http://localhost:4000` 预览)

## ⚡ 常用命令


| 命令 | 描述 |
| :--- | :--- |
| `npm run server` | 启动本地预览 (http://localhost:4000) |
| `npm run build` | 生成静态文件 (到 `public/` 目录) |
| `npm run deploy` | 部署到 GitHub Pages |
| `npm run clean` | 清理缓存 (`db.json` 和 `public/`) |
| `npx hexo new post "..."` | 创建新文章 |

## ✨ 核心功能

* **内容**: 标准 Markdown, LaTeX 数学公式 (by MathJax), 多语言代码高亮
* **主题**: Fluid 响应式设计, 移动端优化, 图片自动懒加载
* **功能**: 集成 Google Analytics, Gitalk 评论系统, 本地搜索

<details>

<summary>📝 点击查看：内容创作与项目结构 (维护者参考)</summary>

### 文章 Front-matter

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

### 项目结构

```text
Yima-Gu.github.io/
├── source/         # 源文件目录 (文章/页面)
├── themes/         # 主题文件
├── public/         # 生成的静态文件
├── _config.yml     # Hexo 主配置
└── _config.fluid.yml # Fluid 主题配置
```
</details>

## 🤝 贡献与支持

如果您发现任何问题或有改进建议，欢迎提交 [Issue](https://github.com/Yima-Gu/Yima-Gu.github.io/issues) 或 [Pull Request](https://github.com/Yima-Gu/Yima-Gu.github.io/pulls)。

⭐ 如果这个项目对您有帮助，请给个 **Star** 支持一下！

