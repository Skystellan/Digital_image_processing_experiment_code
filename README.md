# Digital_image_processing_experiment_code

项目说明
---
这是用于图像处理课程实验的代码仓库，包含若干用于演示相位谱、空域/频域相关与相位谱相关定位方法的脚本。代码主要用 Python + OpenCV/NumPy/Matplotlib 实现，适合在 Windows 下使用 PowerShell 运行。

主要文件
---
- `main_2D.py`：2D 形式的实验主程序，包含相位谱观察、空域/频域/相位谱相关的实现并以 2D 热力图可视化结果。
- `main_3D.py`：3D 形式的实验主程序，和 `main_2D.py` 功能类似，但以 3D 曲面图展示相关结果。
- `main_conbined.py`：将 3D 曲面和 2D 热力图、局部放大图整合到同一页面的可视化脚本。
- `cameraman.tif`, `lena.tif`, `pic1.png`：示例图像（若不存在，脚本会自动生成模拟图像以保证可运行）。

依赖环境
---
建议使用 Python 3.8+（或 3.9/3.10/3.11 均可）。可通过 pip 安装下列常用依赖：

```powershell
pip install numpy opencv-python matplotlib scikit-image
```

（如果使用 conda：
```powershell
conda install numpy matplotlib scikit-image -c conda-forge
pip install opencv-python
```
）

快速开始（PowerShell）
---
1. 进入项目目录：

```powershell
cd 'C:\Users\AherkeSkystellan\Desktop\shuzituxiangchuli'
```

2. 运行 2D 版本演示：

```powershell
python .\main_2D.py
```

3. 运行 3D 版本演示：

```powershell
python .\main_3D.py
```

4. 运行合并视图（combined）：

```powershell
python .\main_conbined.py
```

说明与行为
---
- 所有脚本在启动时会尝试读取当前目录下的标准图像（`cameraman.tif`, `lena.tif`, `pic1.png`）。若找不到这些文件，脚本会用随机或合成图像替代以保证流程可以跑通。
- 主要函数
	- `phase_spectrum_experiment()`：演示相位谱的作用，返回处理后的图像数组用于展示。
	- `spatial_correlation(base_image, template)`：空域归一化互相关（NCC），返回相关图 `rho` 与峰值位置 `(y,x)`。
	- `frequency_correlation(base_image, template)`：频域相关（FFT 实现），返回相关图与峰值位置。
	- `phase_only_correlation(base_image, template)`：仅相位谱相关（POC），返回相关图与峰值位置。

常见问题（FAQ）
---
- 如果运行时报错找不到模块：请确认已安装依赖（见上文 pip 命令）。
- 如果图形窗口没有显示（在某些远程/无头环境）：可以将 `plt.show()` 改为 `plt.savefig('out.png')` 来保存图片后查看。

Git & 贡献
---
- 如果要把修改推到 GitHub：请先 `git add` / `git commit`，然后 `git push`（仓库已配置 ssh/https 按你本地环境）。
- 欢迎通过 Issues 或 Pull Requests 提交改进（请在 PR 描述中说明修改目的与测试方法）。

许可
---
本仓库默认未指定特殊许可证；如果你计划分享或发布，请在仓库根目录添加 `LICENSE` 文件并选择合适的许可证（例如 MIT）。

联系方式
---
如需帮助，可在本仓库打开 Issue 或把运行时错误与输出贴到讨论区，我会继续协助排查。

---
更新记录
---
- 2025-11-18：补充中文 README，说明运行步骤与文件说明；为主要函数添加入口/出口 docstring（在 `main_*.py` 中）。
