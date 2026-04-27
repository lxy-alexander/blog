---
title: "pillow"
published: 2026-04-27
description: "pillow"
image: ""
tags: ["python","pillow"]
category: python
draft: false
lang: ""
createdAt: "2026-04-27T15:07:57.336.640774093Z"
---

# Python Pillow Library

Pillow (PIL Fork) is the most popular ==image processing library (图像处理库) in Python==, supporting opening, editing, and saving (保存) various image formats (图像格式) like JPEG, PNG, BMP, and GIF.

## 1. Installation and Import

Install via `pip install pillow`, but import it as `PIL` (Python Imaging Library) for historical reasons (历史原因).

```python
# pip install pillow
from PIL import Image, ImageDraw, ImageFilter, ImageFont
```

<br>

## 2. Open, Show, Save

`Image.open()` loads an image lazily (惰性加载), and `save()` writes it to disk, with format auto-detected from the extension (扩展名).

```python
from PIL import Image

img = Image.new("RGB", (100, 100), "red")  # create red 100x100 image
print(img.size, img.mode)   # (100, 100) RGB
img.save("red.png")         # save as PNG
# img.show()                # open in default viewer
```

<br>

## 3. Resize and Crop

`resize()` changes dimensions (尺寸), and `crop()` extracts a rectangular region (矩形区域) given as `(left, top, right, bottom)`.

```python
from PIL import Image

img = Image.new("RGB", (200, 200), "blue")
small = img.resize((50, 50))           # resize to 50x50
region = img.crop((0, 0, 100, 100))    # crop top-left quarter
print(small.size, region.size)         # (50, 50) (100, 100)
```

<br>

## 4. Rotate and Flip

`rotate()` rotates by angle (角度), and `transpose()` flips (翻转) horizontally or vertically using flag constants (标志常量).

```python
from PIL import Image

img = Image.new("RGB", (100, 50), "green")
rotated = img.rotate(90, expand=True)              # rotate 90° CCW
flipped = img.transpose(Image.FLIP_LEFT_RIGHT)     # horizontal flip
print(rotated.size, flipped.size)                  # (50, 100) (100, 50)
```

<br>

## 5. Color Mode Conversion

`convert()` changes the color mode (颜色模式), commonly between `RGB`, `L` (grayscale 灰度), and `RGBA` (with alpha channel 透明通道).

```python
from PIL import Image

img = Image.new("RGB", (50, 50), "white")
gray = img.convert("L")       # convert to grayscale
rgba = img.convert("RGBA")    # add alpha channel
print(gray.mode, rgba.mode)   # L RGBA
```

<br>

## 6. Drawing on Images

`ImageDraw` provides drawing tools (绘图工具) to add shapes (形状), lines, and text (文本) onto an image.

```python
from PIL import Image, ImageDraw

img = Image.new("RGB", (200, 100), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((10, 10, 100, 80), outline="red", width=2)
draw.line((0, 0, 200, 100), fill="blue", width=3)
draw.text((50, 40), "Hello", fill="black")
img.save("draw.png")
print("Saved draw.png")  # Saved draw.png
```

<br>

## 7. Apply Filters

`ImageFilter` offers built-in filters (滤镜) like blur (模糊), sharpen (锐化), and edge detection (边缘检测).

```python
from PIL import Image, ImageFilter

img = Image.new("RGB", (100, 100), "gray")
blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
edges = img.filter(ImageFilter.FIND_EDGES)
print(blurred.size, edges.size)  # (100, 100) (100, 100)
```

<br>

## 8. Convert with NumPy

Pillow images can be converted to/from NumPy arrays (数组), enabling integration (集成) with deep learning (深度学习) frameworks.

```python
from PIL import Image
import numpy as np

img = Image.new("RGB", (3, 2), "red")
arr = np.array(img)             # PIL → ndarray
print(arr.shape)                # (2, 3, 3)  → (H, W, C)

img2 = Image.fromarray(arr)     # ndarray → PIL
print(img2.size)                # (3, 2)
```

<br>

## 9. Common Methods Summary

| Method              | Purpose                      |
| ------------------- | ---------------------------- |
| `Image.open(path)`  | Load image (加载图像)        |
| `img.save(path)`    | Save image (保存)            |
| `img.resize((w,h))` | Resize (调整大小)            |
| `img.crop(box)`     | Crop region (裁剪)           |
| `img.rotate(angle)` | Rotate (旋转)                |
| `img.convert(mode)` | Change color mode (转换模式) |
| `img.filter(f)`     | Apply filter (应用滤镜)      |
| `np.array(img)`     | Convert to NumPy (转 NumPy)  |

<br> <br>
