---
title: "Git"
published: 2026-02-03
description: "Git"
image: ""
tags: ["tools","Git"]
category: tools
draft: false
lang: ""
---

# **I. Git — Clean, Fork Sync & Rebase**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> This note covers three related Git workflows: removing tracked and untracked files from the working directory with <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">git clean</code> and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">git restore</code>; keeping a fork up to date with its upstream repository via merge or rebase; and understanding why <strong>rebase is preferred over merge</strong> for fork synchronization.
</div>

---

## 1. Removing Files from the Working Directory

### 1) Remove Git-tracked Files

```bash
git restore .
```

Discards all unstaged changes to tracked files and restores them to the last committed state.

### 2) Remove Untracked Files (`git clean`)

| Command | Danger | Description |
| --- | --- | --- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git clean -f</code> | ⭐ | Deletes a small number of untracked files |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git clean -fd</code> | ⭐⭐ | Also removes untracked directories (e.g., build dirs) |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git clean -fdx</code> | 🔥🔥🔥 | <span style="color:#C0392B;font-weight:600">Removes almost all locally generated content</span>, including files ignored by `.gitignore` |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git clean -fdxn</code> | — | `-n` = dry-run: preview what would be deleted without actually deleting |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Always run <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git clean -fdxn</code> first to preview the files that would be removed before committing to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">-fdx</code>. This operation is <strong>irreversible</strong>.</div>

---

## 2. Updating a Forked Repository

### 1) Method 1: Sync Upstream Locally (Recommended)

#### Step 1: Navigate into Your Local Repository

```bash
cd your-repo
```

#### Step 2: View Existing Remotes

```bash
git remote -v
```

#### Step 3: Add the Upstream Remote (One-time Setup)

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPO.git
```

#### Step 4: Fetch the Latest Changes from Upstream

```bash
git fetch upstream
```

#### Step 5: Integrate Upstream Changes into Your Branch

**Option A — Merge:**

```bash
git checkout main
git merge upstream/main
```

If the upstream default branch is `master`:

```bash
git checkout master
git merge upstream/master
```

**Option B — Rebase (Recommended):**

```bash
git checkout main
git fetch upstream
git rebase upstream/main
```

#### Step 6: Push the Updated Branch to Your Fork

**After merge:**

```bash
git push origin main
```

**After rebase:**

```bash
git push origin main --force-with-lease
```

---

### 2) Method 2: Sync Directly on GitHub (Web UI)

Open your fork on GitHub, then click:

```
Sync fork → Update branch
```

No local commands required.

---

## 3. Why Rebase Is Preferred for Fork Sync

When syncing a personal fork with its upstream, <span style="color:#E8600A;font-weight:700">rebase is the recommended approach</span>.

**Typical scenario:**

- **upstream** has moved ahead with new commits
- Your **fork is behind** upstream
- You have your own local commits on top

![Before rebase or merge](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206214648820)

### 1) Using Merge

- Produces an extra <span style="color:#C0392B;font-weight:600">Merge commit (M)</span>
- History becomes tree-shaped and cluttered
- PRs look noisy and harder to review

![After merge](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206215456002)

### 2) Using Rebase

After rebasing, your original commits are <span style="color:#E8600A;font-weight:700">replayed on top of the upstream</span>. The old commits are discarded from the branch history, and Git creates new commits with the same content but <span style="color:#E8600A;font-weight:700">brand-new commit IDs (D → D')</span>.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> This is why <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--force-with-lease</code> is required after a rebase push. The remote still has the old commit D, while your local branch now has D'. A regular <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git push</code> would be rejected; <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--force-with-lease</code> force-pushes safely by checking that no one else has pushed in the meantime.</div>

Benefits of rebase:

- <span style="color:#2980B9">Clean, linear commit history</span>
- <span style="color:#2980B9">Clearer, easier-to-review PRs</span>
- The **standard practice** for syncing a fork with upstream

![After rebase](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206215538616)

---

## 4. Handling Conflicts in Merge / Rebase / Sync Fork

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Conflict resolution content to be added here.</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git restore .</code> for tracked files and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git clean -fdxn</code> (dry-run first!) for untracked ones; when syncing a fork, always prefer <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">rebase</code> over <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">merge</code> to keep a clean, linear history — then <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">push --force-with-lease</code> to update the remote safely.</div>









