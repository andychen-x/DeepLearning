# GitHub 入门学习指南

> 从零开始，系统掌握 Git 与 GitHub 的核心用法。

---

## 目录

1. [Git vs GitHub：先搞清这个区别](#1-git-vs-github先搞清这个区别)
2. [必学的六个命令](#2-必学的六个命令)
3. [三个核心区域](#3-三个核心区域)
4. [动手：完成你的第一个仓库](#4-动手完成你的第一个仓库)
5. [分支：协作的基础](#5-分支协作的基础)
6. [GitHub Pages：搭建免费网站](#6-github-pages搭建免费网站)
7. [学会看开源仓库](#7-学会看开源仓库)
8. [常见坑与解决方法](#8-常见坑与解决方法)

---

## 1. Git vs GitHub：先搞清这个区别

很多人把这两个混为一谈，但它们完全不同。

| | Git | GitHub |
|--|--|--|
| **是什么** | 版本控制工具（装在本地） | 托管平台（网站） |
| **能否离线使用** | ✅ 可以 | ❌ 不行 |
| **主要作用** | 记录代码每次变更 | 把代码放到云端，支持协作 |
| **额外功能** | 无 | Issue、PR、Actions、Pages 等 |

**类比理解：**
- Git 是相机，负责「拍照」（记录代码快照）
- GitHub 是相册平台，负责「展示和分享」（托管与协作）

> ⚠️ **建议：先学 Git，再学 GitHub。** 不理解 Git 的底层逻辑，遇到冲突、分支等问题时会完全懵掉。

---

## 2. 必学的六个命令

日常开发 90% 的场景只需要这六个命令，其余的用到再查。

```bash
git init          # 在当前目录初始化一个新的 Git 仓库
git add <文件>    # 将文件变更加入暂存区
git commit        # 将暂存区内容保存为一个版本
git push          # 将本地版本推送到远程仓库（如 GitHub）
git pull          # 将远程最新版本拉取到本地
git clone <URL>   # 将 GitHub 上的仓库完整复制到本地
```

---

## 3. 三个核心区域

理解这个模型，能消除 Git 八成的困惑。

```
工作区  ──git add──▶  暂存区  ──git commit──▶  本地仓库  ──git push──▶  远程仓库
```

| 区域 | 说明 |
|--|--|
| **工作区** | 你直接编辑文件的地方 |
| **暂存区** | 确认「这次要提交哪些改动」的中间层 |
| **本地仓库** | 保存在本机的完整版本历史 |
| **远程仓库** | GitHub 上的云端版本 |

**为什么要有暂存区？**

假设你同时改了 5 个文件，但只有 3 个改动属于同一件事，另外 2 个属于下一个任务。暂存区让你灵活选择「这次 commit 包含哪些改动」，而不是一股脑全提交。

---

## 4. 动手：完成你的第一个仓库

### 4.1 安装 Git

| 系统 | 方法 |
|--|--|
| Windows | 去 [git-scm.com](https://git-scm.com) 下载安装包 |
| macOS | 终端运行 `git --version`，有提示则已自带 |
| Linux | `sudo apt install git` 或对应包管理器 |

### 4.2 初始配置

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

这两项信息会出现在每条 commit 记录里，让团队知道是谁提交的。

### 4.3 创建并推送第一个仓库

**在 GitHub 上创建仓库：**

1. 点击右上角 `+` → `New repository`
2. 填写仓库名（如 `my-first-repo`）
3. 勾选 `Add a README file`
4. 点击 `Create repository`

**克隆到本地，完成一次完整的工作流：**

```bash
# 克隆仓库到本地
git clone https://github.com/你的用户名/my-first-repo.git
cd my-first-repo

# 创建一个新文件
echo "Hello GitHub" > hello.txt

# 查看当前状态
git status

# 加入暂存区
git add hello.txt

# 提交，-m 后面是本次提交的说明
git commit -m "添加了 hello.txt"

# 推送到 GitHub
git push
```

刷新 GitHub 页面，看到 `hello.txt` 了吗？

🎉 **恭喜！你完成了 Git 工作流的完整闭环。** 后续无论多复杂的操作，都是这个基本流程的变体。

---

## 5. 分支：协作的基础

### 什么是分支？

分支相当于「平行时间线」。你可以在新分支上随意折腾，成功了合并回主线，失败了直接丢弃，完全不影响原有代码。

### 基本操作

```bash
# 创建新分支并切换过去
git checkout -b new-feature

# 在分支上做改动并提交
echo "new stuff" > feature.txt
git add feature.txt
git commit -m "尝试新功能"

# 切回主分支（会发现 feature.txt 不见了，它只在新分支上）
git checkout main

# 如果功能 OK，将新分支合并到主分支
git merge new-feature
```

### GitHub 协作的标准流程（GitHub Flow）

```
Fork 仓库 → 创建分支 → 修改代码 → 提 Pull Request → 维护者 Review → 合并
```

这是所有开源项目协作的基础流程。

---

## 6. GitHub Pages：搭建免费网站

用实际产出来巩固学习效果。GitHub Pages 可以免费将仓库变成一个网站，无需服务器，无需付费。

**最简单的做法：**

1. 创建一个仓库，名字必须是 `你的用户名.github.io`
2. 在仓库中放一个 `index.html` 文件
3. 几分钟后访问 `https://你的用户名.github.io`

你就有了一个真实可访问的个人网站。这会让你有动力继续探索：怎么让页面更好看？怎么加博客？怎么自动部署？每个问题都会引你学到新东西。

---

## 7. 学会看开源仓库

GitHub 上最有价值的资源，是别人的代码和项目运作方式。

### 推荐给初学者的仓库

| 仓库 | 推荐理由 |
|--|--|
| [github/docs](https://github.com/github/docs) | GitHub 官方文档，结构清晰，适合学习目录组织 |
| [firstcontributions/first-contributions](https://github.com/firstcontributions/first-contributions) | 专门引导新手完成第一次 PR，有详细图文教程，10 分钟走完 |
| [EbookFoundation/free-programming-books](https://github.com/EbookFoundation/free-programming-books) | 看它的 README 如何组织海量信息 |

> 💡 **强烈推荐先去 `first-contributions`。** 它存在的唯一目的就是让你完成第一次 PR。走完整个流程后，你就不再是旁观者了。

**看仓库时重点关注：**
- 目录结构怎么组织
- README 怎么写
- Issue 和 PR 如何管理
- `.github/workflows` 里的自动化配置（CI/CD）

---

## 8. 常见坑与解决方法

### 坑 1：Detached HEAD

```
You are in 'detached HEAD' state.
```

**原因：** checkout 到了某个具体的 commit，而不是分支。

**解决：** 执行 `git checkout main` 回到主分支即可。

> ⚠️ 在 detached HEAD 状态下做的 commit，如果没有创建新分支，很难找回。

---

### 坑 2：推送被拒绝

```
! [rejected] main -> main (fetch first)
```

**原因：** 远程有你本地没有的更新。

**解决：**
```bash
git pull        # 先拉取远程变更，处理可能的冲突
git push        # 再推送
```

> ❌ **不要用 `git push -f`（force push）！** 这会强制用本地版本覆盖远程，在协作项目中等于删掉别人的工作成果。

---

### 坑 3：提交了不该提交的文件

**解决：** 在仓库根目录创建 `.gitignore` 文件，将不想追踪的文件写进去。

```gitignore
# 示例 .gitignore
node_modules/
*.log
.env
dist/
__pycache__/
```

各语言和框架的 `.gitignore` 模板可以直接从 [github/gitignore](https://github.com/github/gitignore) 取用。

---

## 快速参考卡

```bash
# === 初始化 ===
git init                          # 初始化仓库
git clone <URL>                   # 克隆远程仓库

# === 日常提交 ===
git status                        # 查看当前状态
git add <文件>                    # 暂存指定文件
git add .                         # 暂存所有改动
git commit -m "说明"              # 提交并附说明

# === 同步远程 ===
git push                          # 推送到远程
git pull                          # 拉取远程更新

# === 分支操作 ===
git branch                        # 查看所有分支
git checkout -b <分支名>          # 创建并切换新分支
git checkout <分支名>             # 切换分支
git merge <分支名>                # 合并指定分支到当前分支

# === 配置 ===
git config --global user.name "名字"
git config --global user.email "邮箱"
```

---

*遇到问题，先 `git status` 看看当前状态，再根据提示处理。90% 的问题，Git 自己会告诉你怎么解决。*
