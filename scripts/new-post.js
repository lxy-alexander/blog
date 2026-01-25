/* Create a new post markdown file with front-matter */

import fs from "fs";
import path from "path";

function getDate() {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, "0");
  const day = String(today.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function removeExt(name) {
  return name.replace(/\.(md|mdx)$/i, "");
}

const args = process.argv.slice(2);

if (args.length === 0) {
  console.error(`Error: No filename argument provided
Usage: npm run new -- <filename>`);
  process.exit(1);
}

// 支持空格 + 子目录
let inputName = args.join(" ").trim();

// Add .md extension if not present
const fileExtensionRegex = /\.(md|mdx)$/i;
if (!fileExtensionRegex.test(inputName)) {
  inputName += ".md";
}

// ✅ 写到当前执行目录
const cwd = process.env.INIT_CWD || process.cwd();
const fullPath = path.resolve(cwd, inputName);

if (fs.existsSync(fullPath)) {
  console.error(`Error: File ${fullPath} already exists`);
  process.exit(1);
}

// 创建目录
const dirPath = path.dirname(fullPath);
if (!fs.existsSync(dirPath)) {
  fs.mkdirSync(dirPath, { recursive: true });
}

// ========== tags 生成 ==========
// 文件名 tag
const fileBaseName = removeExt(path.basename(inputName));

// 当前文件父目录：绝对路径
const fileParentAbs = path.dirname(fullPath);

// 切路径片段
const parts = fileParentAbs.split(path.sep).filter(Boolean);

// 找到 src/content/posts 位置
let tagDirs = [];
for (let i = 0; i < parts.length - 2; i++) {
  if (parts[i] === "src" && parts[i + 1] === "content" && parts[i + 2] === "posts") {
    // 只取 posts 后面的部分当 tags 目录
    tagDirs = parts.slice(i + 3);
    break;
  }
}

// 如果路径里不含 src/content/posts，为了“不变长”，只取最后 2 层目录
if (tagDirs.length === 0) {
  tagDirs = parts.slice(-2);
}

// tags = 目录tag + 文件名tag
const tags = [...tagDirs, fileBaseName];

const content = `---
title: "${fileBaseName}"
published: ${getDate()}
description: "${fileBaseName}"
image: ""
tags: ${JSON.stringify(tags)}
category: ""
draft: false
lang: ""
---
`;

fs.writeFileSync(fullPath, content, "utf8");

console.log(`✅ Post created: ${fullPath}`);
