# ROpenCV：R 与 OpenCV 之间的桥梁 · 基础教程

**在线阅读**：<https://judy-zq.github.io/ROpenCV-Tutorial/>

一本面向 R 用户的 OpenCV 入门教程。全书以可运行的示例为主线，从图像矩阵的内存表示讲起，
依次覆盖核心模块、图像处理、摄像机标定、特征匹配与视频分析，共五篇、十二章。
每章开头有「本章导读」说明目标与阅读路线，章末有「本章小结」汇总关键接口与易错点。

| 篇 | 主题 | 内容 |
|----|------|------|
| 一 | 背景与基础 | R 语言梗概、图像处理工具现状、OpenCV 简介、R 语法速查 |
| 二 | ROpenCV 入门 | 封装机制、图像矩阵 `cvMat`、图像的读取与显示 |
| 三 | 核心模块 | 像素遍历、查表、边缘扩展、掩码、通道与频域运算 |
| 四 | 图像处理 | 色彩与几何变换、形态学、边缘检测、直方图、轮廓与分割 |
| 五 | 进阶模块 | 摄像机标定、特征检测与匹配、视频读写与运动分析 |

书后附有[术语与符号约定](https://judy-zq.github.io/ROpenCV-Tutorial/12-glossary.html)，
汇总 BGR 通道顺序、0/1 起始下标换算等全书反复出现的约定。

## 关于本仓库

这个仓库**只保存渲染结果**（`_book/`），打开仓库即可看到全部 HTML 与插图；
推送到 `main` 分支会由 `.github/workflows/pages.yml` 自动部署到 GitHub Pages。
Quarto 源文件（`.qmd`）、配置与素材保存在本地源项目中，不随本仓库发布，
因此书中不提供「编辑本页」入口。

配套代码：[ROpenCV](https://github.com/Judy-ZQ/ROpenCV)（R 与 OpenCV 的桥接包）。

## 本地构建（维护者）

源项目需要 R、[ROpenCV](https://github.com/Judy-ZQ/ROpenCV) 与 Quarto（>= 1.4）：

```bash
quarto render       # 渲染到 _book/
quarto preview      # 本地预览
```

或 `make html` / `make preview` / `make clean`。部分示例会弹出图像窗口或读取视频文件，
渲染时需要可运行 ROpenCV 的桌面环境。若 Quarto 不在 PATH 中，RStudio 自带一份
（Windows 下通常为 `D:/RStudio/resources/app/bin/quarto/bin/quarto.cmd`）。

`quarto render` 结束后会自动执行 `scripts/post-render.R`：裁掉 R 图形设备留下的
白边、清理历次渲染累积的孤儿图、移除图片的内联宽度、补 `_book/.nojekyll`。
裁白边依赖可选的 R 包 **magick**；未安装时会跳过这一步，其余构建不受影响。
请始终渲染整本书：单独渲染某个 `.qmd` 不会触发项目级后处理，该章图片会保留白边。

本地源项目的主要文件：

```
├── _quarto.yml              # 分篇、主题、字体、代码块与图片样式
├── index.qmd                # 封面页：全书结构、前言、热身示例、致谢
├── 01-introduction.qmd      # 第 1 章 背景与概览
├── 02-rbasis.qmd            # 第 2 章 R 语言基础
├── 03-ropencv-mechanism.qmd # 第 3 章 ROpenCV 的封装机制
├── 04-imagebasis.qmd        # 第 4 章 图像矩阵 cvMat
├── 05-firstshow.qmd         # 第 5 章 图像的读取、创建与显示
├── 06-coremodule.qmd        # 第 6 章 核心模块：像素级运算
├── 07-imgproc-1.qmd         # 第 7 章 色彩、几何变换与形态学
├── 07-imgproc-2.qmd         # 第 8 章 金字塔、边缘检测与直方图
├── 07-imgproc-3.qmd         # 第 9 章 模板匹配、轮廓与图像分割
├── 08-calib3d.qmd           # 第 10 章 摄像机标定与单应性
├── 09-feature2d.qmd         # 第 11 章 特征检测与匹配
├── 10-video.qmd             # 第 12 章 视频读写与运动分析
├── 11-postscript.qmd        # 后记
├── 12-glossary.qmd          # 附录：术语与符号约定
├── style/custom.scss        # 封面、字体、代码块、图片尺寸与居中
├── scripts/post-render.R    # 渲染后处理：裁白边、清孤儿图、补 .nojekyll
└── images/ video/ data/     # 章节素材
```

> 章号由 Quarto 按章节顺序生成，与文件名前缀不完全一致；
> 三个 `07-imgproc-*.qmd` 同属第四篇「图像处理」，依次是第 7～9 章。

## 许可

正文内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可。
