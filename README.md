**LBH_Image_nodes Node Package Function Introduction**

### Functional Modules
1. **Image Saving and Output**
    - **JPG Save Node**
        - **Basic Version**: Supports custom path templates, automatic date naming, and fixed quality (90) saving. Features include automatic subdirectory creation, file name conflict resolution, and EXIF metadata recording.
        - **Advanced Version**: Adjustable quality parameters (1-100) and support for embedding seed values in metadata.
    - **Proportional Scaling Node**: Automatically maintains aspect ratio based on the longest side, supports multiple interpolation algorithms (Lanczos, bicubic, bilinear, nearest-neighbor), and allows precise control of output size (64-8192, step size 8).
2. **Advanced Block Processing**
    - **Image/Video Block Splitter**: Supports block processing of single frames and video sequences, adjustable overlap ratio (5%-30%), block magnification function can automatically fill insufficient areas or smartly magnify blocks, and outputs block coordinate information for subsequent stitching.
    - **Smart Stitcher**: Gaussian/linear edge blending algorithms, supports restoring original size or maintaining magnified size, automatic frame-by-frame stitching for video sequences, and has color reference matching functions (automatic exposure adjustment).
3. **Precision Cropping and Blending**
    - **Crop Data Generation**: Can manually specify crop region coordinates and dimensions, has face detection cropping function (calculates face region through template matching), and outputs standardized CROP_DATA format.
    - **Image Paste Blending**: Has edge feathering with adjustable transparency blending intensity (0-1), sharpening control (0-3 levels), synchronously outputs blending masks, and can automatically handle edge transitions to eliminate seams.
4. **Utility Tools**
    - **Random Seed Generator**: Supports fixed seeds or automatic random generation, and can automatically set PyTorch random seeds to ensure reproducibility.
    - **Size Input Node**: Standardized size parameter input (independent control of width and height), with range limits (64-8192) and step size settings (multiples of 8).

### Technical Highlights
- Video processing capability: Block/stitching splitters natively support batch processing of video sequences.
- Lossless workflow: All processing nodes retain original data for subsequent adjustments.
- Professional blending algorithms: Lanczos interpolation, Gaussian weight blending, adaptive exposure matching.
- Metadata support: EXIF records processing parameters for traceability.
- Error defense: Automatic size alignment, boundary checking, and null value handling.

### Typical Application Scenarios
- High-definition image enlargement: Block processing → Local enhancement → Seamless stitching.
- Face editing: Automatic positioning → Local repair → Natural blending.
- Video enhancement: Frame-by-frame block processing → Temporal consistency maintenance.
- Dataset creation: Precision cropping + Standardized saving.

All nodes have been optimized to support ComfyUI's batch processing and workflow orchestration. The Chinese interface prompts make operations more intuitive. It is recommended to build a stable production pipeline through the combination of "random seed + size input + block processing + intelligent saving".


**LBH_Image_nodes 节点包功能简介**

### 功能模块
1. **图像保存与输出**
    - **JPG保存节点**
        - **基础版**：支持自定义路径模板、自动日期命名、固定质量(90)保存；特色为自动创建子目录、文件名冲突处理、EXIF元数据记录。
        - **高级版**：可调质量参数(1-100)、支持seed值嵌入元数据。
    - **等比例缩放节点**：按最长边自动保持比例缩放，支持多种插值算法（Lanczos/双三次/双线性/最近邻），可精确控制输出尺寸（64-8192，步长8）。
2. **高级分块处理**
    - **图像/视频分块器**：支持单帧和视频序列分块处理，可调重叠比例(5%-30%)，分块放大功能可自动填充不足区域或智能放大分块，输出分块坐标信息供后续拼接使用。
    - **智能拼接器**：高斯/线性边缘混合算法，支持恢复原始尺寸或保持放大尺寸，视频序列逐帧自动拼接，有颜色参考匹配功能（自动曝光调整）。
3. **精准裁剪与融合**
    - **裁剪数据生成**：可手动指定裁剪区域坐标和尺寸，有人脸自动检测裁剪功能（通过模板匹配计算人脸区域），输出标准化CROP_DATA格式。
    - **图像粘贴融合**：带透明度混合的边缘羽化（可调强度0-1），有锐化控制（0-3级），同步输出融合遮罩，能自动处理边缘过渡消除接缝。
4. **实用工具**
    - **随机种子生成器**：支持固定种子或自动随机生成，可自动设置PyTorch随机种子确保可复现性。
    - **尺寸输入节点**：可标准化尺寸参数输入（宽/高独立控制），有范围限制（64-8192）和步长设置（8的倍数）。

### 技术亮点
- 视频处理能力：分块/拼接器原生支持视频序列批处理。
- 无损工作流：所有处理节点保留原始数据供后续调整。
- 专业级混合算法：Lanczos插值、高斯权重混合、自适应曝光匹配。
- 元数据支持：EXIF记录处理参数，便于溯源。
- 错误防御：自动尺寸对齐、边界检查、空值处理。

### 典型应用场景
- 高清图像放大：分块处理→局部增强→无缝拼接。
- 人脸编辑：自动定位→局部修复→自然融合。
- 视频增强：逐帧分块处理→时域一致性保持。
- 数据集制作：精准裁剪+标准化保存。

所有节点都经过优化以支持ComfyUI的批处理和流程编排，有中文界面提示使操作更直观，建议通过「随机种子+尺寸输入+分块处理+智能保存」组合构建稳定生产管线。
