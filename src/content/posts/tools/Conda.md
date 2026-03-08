---
title: "Conda"
published: 2026-02-03
description: "Conda"
image: ""
tags: ["tools","Conda"]
category: tools
draft: false
lang: ""
---

# **I. Conda Cache Cleanup — Free Up Disk Space**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> Conda accumulates downloaded packages, extracted tarballs, and index caches over time. Cleaning these up frees disk space and reduces environment cruft — without touching any of your existing environments.
</div>

---

## 1. Check Conda Cache Usage (Optional)

```bash
conda info
conda config --show pkgs_dirs
du -sh ~/miniconda3/pkgs 2>/dev/null
```

---

## 2. One-command Cache Cleanup (Recommended)

```bash
conda clean -a -y
```

| Flag | Meaning |
| --- | --- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">-a</code> | **all** — cleans every type of cache and leftover |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">-y</code> | **yes** — auto-confirms without prompting |

This removes:

- Downloaded `.tar.bz2` / `.conda` package archives
- Extracted package caches
- Index caches
- Unused package caches

---

## 3. Also Clean pip Cache (If Used Inside Conda Envs)

```bash
pip cache purge
```

---

## 4. Verify the Space Was Freed (Optional)

```bash
du -sh ~/miniconda3/pkgs 2>/dev/null
```

---

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">conda clean -a</code> <strong>does not delete your environments</strong> — it only removes cached downloads. The trade-off is that the next time you install a package, conda may need to re-download it.</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Run <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">conda clean -a -y</code> to safely reclaim disk space from package caches — your environments stay completely intact.</div>