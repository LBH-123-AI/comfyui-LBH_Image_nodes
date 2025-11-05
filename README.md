**LBH_Image_nodes_ComfyUI 节点包功能简介**

### 功能模块
1. **图像保存与输出 (Image Save & Output)**
    - **JPG_Save (保存JPG图像)**
       - **基础版**: 支持自定义路径模板、自动日期命名、固定质量(95)保存；特色功能包括自动创建子目录、文件名冲突处理、EXIF元数据记录
       - **高级版**: 可调节质量参数(1-100)、支持seed值嵌入元数据、更完善的元数据记录
    - **ImageResizer (等边缩放)**: 按最长边自动保持比例缩放，支持多种插值算法(Lanczos/双三次/双线性/最近邻)，精确控制输出尺寸(64-8192，步长8)
2. **高级分块处理 (Advanced Tiling Processing)**
    - **ImageTileSplitter (图像/视频分块器)**: 支持单帧和视频序列分块处理，可调重叠比例(5%-30%)，支持按tile分组输出，自动填充不足区域
    - **NaturalTileMerger (图像/视频拼接器)**: 高斯/线性边缘混合算法，支持颜色参考匹配和自动曝光调整，视频序列逐帧智能拼接
3. **精准裁剪与检测 (Precise Cropping & Detection)**
    - **WAS_Image_Crop_Data (裁剪数据生成)**: 手动指定裁剪区域坐标和尺寸，生成标准化CROP_DATA格式
    - **CropDataFromFace (人脸裁剪数据提取)**: 通过模板匹配自动检测人脸区域并计算裁剪坐标
    - **FaceDetectorWithCrop (人脸检测+裁剪)**: 集成人脸检测器，支持阈值调节、区域扩张和裁剪因子控制，输出分割结果和裁剪人脸
4. **实用工具 (Utility Tools)**
    - **RandomSeedNode (随机种子)**: 支持固定种子或自动随机生成，自动设置PyTorch随机种子确保可复现性
    - **SizeInput (尺寸输入)**: 标准化尺寸参数输入，独立控制宽高，范围限制(64-8192)，步长8的倍数
5. **视频处理专项 (Video Processing Specialized)**
    - **UltraStitchTiler (UltraStitch视频分段)**: 专为长视频设计的10段+翻页分块系统，支持重叠帧数和分段起始点控制
    - **UltraStitchMerger (UltraStitch视频拼接)**: 智能调色匹配和多种叠化曲线(余弦/正弦/立方/线性)，消除片段间颜色和亮度差异


**LBH_Image_nodes_ComfyUI Package Overview**

### Functional Modules
1. **Image Save & Output**
    - **JPG_Save Nodes**
       - **Basic Version**: Custom path templates, auto-date naming, fixed quality(95); features: auto subdirectory creation, filename conflict handling, EXIF metadata
       - **Advanced Version**: Adjustable quality(1-100), seed embedding in metadata, enhanced metadata recording
    - **ImageResizer**: Aspect ratio preservation by longest side, multiple interpolation algorithms (Lanczos/bicubic/bilinear/nearest), precise size control (64-8192, step 8)
2. **Advanced Tiling Processing**
    - **ImageTileSplitter**: Single frame and video sequence tiling, adjustable overlap ratio(5%-30%), tile-grouped output for videos, automatic padding
    - **NaturalTileMerger**: Gaussian/linear edge blending, color reference matching with auto-exposure adjustment, frame-by-frame video stitching
3. **Precise Cropping & Detection**
    - **WAS_Image_Crop_Data**: Manual crop region specification with standardized CROP_DATA format output
    - **CropDataFromFace**: Automatic face region detection via template matching for crop coordinates
    - **FaceDetectorWithCrop**: Integrated face detector with threshold control, region dilation, and crop factor adjustment
4. Utility Tools
    - **RandomSeedNode**: Fixed or auto-random seed generation with PyTorch seed setting for reproducibility
    - **SizeInput**: Standardized dimension input with independent width/height control (64-8192 range, step 8)
5. **Video Processing Specialized**
    - **UltraStitchTiler**: 10-segment + pagination system for long videos, overlap frame control, and segment starting point adjustment
    - **UltraStitchMerger**: Intelligent color matching with multiple blending curves (cosine/sine/cubic/linear), seamless video segment integration
