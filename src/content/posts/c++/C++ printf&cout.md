---
title: "C++ printf&cout"
published: 2026-03-07
description: "C++ printf&cout"
image: ""
tags: ["c++","C++ printf&cout"]
category: c++
draft: false
lang: ""
---

# **I. `printf` vs `cout` (输出方式选择)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">std::cout</code> is the modern C++ default — <span style="color:#E8600A;font-weight:700">type-safe (类型安全)</span> and integrates with C++ objects. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">printf</code> is a C-style function useful for strict formatting control and C API interop.
</div>

---

## <span style="color:#E8600A">1.</span> **When to use each (使用场景)**

| | `std::cout` | `printf` |
|---|---|---|
| Type safety (类型安全) | ✅ Yes | ❌ No (format string mismatch crashes) |
| C++ objects / templates | ✅ Natural | ❌ Not supported |
| Strict format control (格式控制) | Verbose | ✅ Concise (`%d`, `%.2f`) |
| C API / system code (系统代码) | ❌ Awkward | ✅ Preferred |

---

## <span style="color:#E8600A">2.</span> **Recommendation (推荐)**

### 1) Default: use `std::cout`

```cpp
std::cout << "value: " << x << "\n";
```

### 2) Use `printf` for formatting

```cpp
printf("min: %d, avg: %.2f\n", mn, avg);
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Mixing <code>printf</code> and <code>cout</code> in the same program can cause <span style="color:#C0392B;font-weight:600">output ordering issues (输出顺序问题)</span> if you forget to call <code>ios::sync_with_stdio(false)</code>.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">std::cout</code> by default; reach for <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">printf</code> only when you need precise format strings or are working with C code.
</div>