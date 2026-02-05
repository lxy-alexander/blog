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





# Update a Forked Repository

## Method 1: Sync the upstream repository to your fork (Recommended)

### Step 1: Go to your local repo

```bash
cd your-repo
```

### Step 2: Check existing remotes

```bash
git remote -v
```

### Step 3: Add the upstream remote (only once)

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPO.git
```

### Step 4: Fetch the latest changes from upstream

```bash
git fetch upstream
```

### Step 5: Merge upstream changes into your branch

If the upstream branch is `main`:

```bash
git checkout main
git merge upstream/main
```

If the upstream branch is `master`:

```bash
git checkout master
git merge upstream/master
```

### Step 6: Push the updated branch to your fork

```bash
git push origin main
```

------

## Method 2: Update using rebase (Clean history)

```bash
git checkout main
git fetch upstream
git rebase upstream/main
git push origin main --force-with-lease
```

------

## Method 3: Update directly on GitHub (Web UI)

### Step 1: Open your fork on GitHub

Go to your fork repository page.

### Step 2: Click the sync button

Click:
`Sync fork` → `Update branch`

------

## Common issues

### Merge conflicts

If conflicts happen, fix them manually, then run:

```bash
git add .
git commit
git push
```
