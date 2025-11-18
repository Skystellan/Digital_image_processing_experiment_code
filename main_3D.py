import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import transform
import matplotlib
import platform
import os

# ==================== 字体设置（解决中文显示问题） ====================
def setup_chinese_font():
    """
    自动检测并设置中文字体

    入口 (无):
        无需外部参数，函数会根据运行平台自动查找常见中文字体并尝试加载。

    出口 (无):
        无返回值。函数会设置 matplotlib 的字体配置（全局影响），并在控制台打印加载结果或警告。
    """
    system = platform.system()
    
    # 常见中文字体路径
    font_paths = []
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf",
        ]
    elif system == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
    
    # 查找可用的字体
    font_found = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                matplotlib.font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = [matplotlib.font_manager.FontProperties(fname=font_path).get_name()]
                font_found = True
                print(f"已加载中文字体: {font_path}")
                break
            except:
                continue
    
    if not font_found:
        print("警告：未找到中文字体，将使用默认字体（可能无法显示中文）")
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

setup_chinese_font()

# ==================== 1. 相位谱观察实验 ====================
def phase_spectrum_experiment():
    """
    实验1：观察相位谱的作用（部分频谱恢复 + 交换相位谱）

    入口:
        无（函数内部会尝试读取工作目录下的 'cameraman.tif' 和 'lena.tif'，若不存在会生成模拟图像）。

    出口:
        返回两个 numpy.ndarray：
            I1: 基准图像（浮点型，shape (M,N)），用于后续实验或显示。
            I3: 缩放后的第二张图像（浮点型，shape 根据缩放而定）。

    说明:
        函数同时会显示并保存（显示）几幅图像用于可视化实验结果（不做文件写出）。
    """
    print("="*50)
    print("实验1：相位谱作用观察")
    print("="*50)
    
    # 读取图像
    try:
        I1 = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
        I2 = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)
        if I1 is None or I2 is None:
            raise FileNotFoundError("无法读取图像文件")
    except:
        print("警告：未找到标准图像文件，生成模拟图像...")
        I1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        I2 = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    # 将I2缩小0.5倍
    scale = 0.5
    I3 = transform.resize(I2, (int(I2.shape[0]*scale), int(I2.shape[1]*scale)), 
                         anti_aliasing=True, preserve_range=True).astype(np.uint8)
    
    I1 = I1.astype(np.float64)
    I3 = I3.astype(np.float64)
    
    # 显示原始图像
    plt.figure(figsize=(20, 5))
    plt.subplot(141), plt.imshow(I1, cmap='gray'), plt.title('基准图 I1')
    plt.subplot(142), plt.imshow(I3, cmap='gray'), plt.title('缩放图 I3')
    
    # FFT变换
    F1 = np.fft.fftshift(np.fft.fft2(I1))
    F3 = np.fft.fftshift(np.fft.fft2(I3))
    
    # 实验1.1：部分频谱恢复
    h, w = 60, 60
    Ft1 = np.zeros_like(F1, dtype=complex)
    Ft1[21:h+21, 50:w+50] = F1[80:h+80, 50:w+50]
    
    II1 = np.real(np.fft.ifft2(np.fft.ifftshift(Ft1)))
    II1 = II1 / np.max(II1)
    
    # 实验1.2：交换相位谱
    marginA = np.abs(F1)
    phaseB = np.angle(F3)
    Ft2 = marginA * np.exp(1j * phaseB)
    II2 = np.real(np.fft.ifft2(np.fft.ifftshift(Ft2)))
    II2 = II2 / np.max(II2)
    
    plt.subplot(143), plt.imshow(II2, cmap='gray'), plt.title('交换相位谱结果')
    
    # 显示部分频谱恢复结果
    plt.subplot(144), plt.imshow(II1, cmap='gray'), plt.title('部分频谱恢复')
    plt.tight_layout()
    plt.show()
    
    print("相位谱实验完成！")
    return I1, I3

# ==================== 2. 空域相关计算 ====================
def spatial_correlation(base_image, template):
    """
    空域归一化互相关（NCC）计算。

    入口:
        base_image (numpy.ndarray): 基准图像，二维灰度数组，shape (m, n)。
        template (numpy.ndarray): 待匹配的子图（模板），二维灰度数组，shape (h, w)，h<=m, w<=n。

    出口:
        rho (numpy.ndarray): 大小为 (m-h+1, n-w+1) 的相关系数图，每个元素为模板与该位置窗口的归一化互相关系数，取值范围近似在 [-1,1]。
        peak_pos (tuple): 峰值位置索引 (y, x)，表示 rho 中最大值的位置，即模板最佳匹配处的坐标（以 rho 的坐标系为准，注意为 (row, col)）。

    注意:
        - 当窗口能量为 0 时，相关系数处以 0 处理以避免除零。
    """
    print("执行空域相关计算...")
    
    m, n = base_image.shape
    h, w = template.shape
    
    rho = np.zeros((m - h + 1, n - w + 1), dtype=np.float64)
    template_energy = np.sqrt(np.sum(template ** 2))
    
    for y in range(m - h + 1):
        for x in range(n - w + 1):
            window = base_image[y:y+h, x:x+w]
            numerator = np.sum(template * window)
            denominator = template_energy * np.sqrt(np.sum(window ** 2))
            rho[y, x] = numerator / denominator if denominator != 0 else 0
    
    peak_idx = np.unravel_index(np.argmax(rho), rho.shape)
    return rho, peak_idx

# ==================== 3. 频域相关计算 ====================
def frequency_correlation(base_image, template):
    """
    频域相关计算（基于 FFT 的加速实现）。

    入口:
        base_image (numpy.ndarray): 基准图像，二维灰度数组，shape (m, n)。
        template (numpy.ndarray): 模板图像，二维灰度数组，shape (h, w)，会被零填充到与 base_image 相同大小后做 FFT。

    出口:
        correlation_map (numpy.ndarray): 实部相关映射，大小与 base_image 相同，峰值位置即为模板的最佳匹配处（以 base_image 的坐标系为准）。
        peak_pos (tuple): 峰值位置索引 (y, x)，表示 correlation_map 中最大值处的索引（row, col）。

    说明:
        该方法利用频域的相关定理：IFFT(FFT(base) * conj(FFT(template_padded))) 得到相关结果。
    """
    print("执行频域相关计算...")
    
    m, n = base_image.shape
    h, w = template.shape
    
    # 子图延拓
    template_padded = np.zeros_like(base_image, dtype=np.float64)
    template_padded[:h, :w] = template
    
    # FFT
    F_base = np.fft.fft2(base_image)
    F_template = np.fft.fft2(template_padded)
    
    # 相关定理
    correlation_freq = F_base * np.conj(F_template)
    correlation_map = np.real(np.fft.ifft2(correlation_freq))
    
    peak_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    return correlation_map, peak_idx

# ==================== 4. 仅相位谱相关计算 ====================
def phase_only_correlation(base_image, template):
    """
    仅相位谱相关（POC, Phase-Only Correlation）。

    入口:
        base_image (numpy.ndarray): 基准图像，二维灰度数组，shape (m, n)。
        template (numpy.ndarray): 模板图像，二维灰度数组，shape (h, w)，将被零填充到与 base_image 相同大小。

    出口:
        poc_map (numpy.ndarray): 相位相关映射（实部），峰值对应模板位置，大小与 base_image 相同。
        peak_pos (tuple): 峰值位置索引 (y, x)，表示 poc_map 中最大值的索引（row, col）。

    说明:
        该方法只保留频域的相位信息（将幅度置为 1），常用于对位移不敏感且对相位信息敏感的匹配场景。
    """
    print("执行仅相位谱相关计算...")
    
    m, n = base_image.shape
    h, w = template.shape
    
    template_padded = np.zeros_like(base_image, dtype=np.float64)
    template_padded[:h, :w] = template
    
    F_base = np.fft.fft2(base_image)
    F_template = np.fft.fft2(template_padded)
    
    # 仅保留相位信息
    F_base_phase = np.exp(1j * np.angle(F_base))
    F_template_phase = np.exp(1j * np.angle(F_template))
    
    correlation_freq = F_base_phase * np.conj(F_template_phase)
    poc_map = np.real(np.fft.ifft2(correlation_freq))
    
    peak_idx = np.unravel_index(np.argmax(poc_map), poc_map.shape)
    return poc_map, peak_idx

# ==================== 5. 主实验流程 ====================
def main_experiment():
    """
    主实验流程：对比空域、频域与仅相位谱三种方法的子图定位结果，并可视化。

    入口:
        无直接参数。函数内部会尝试读取 `pic1.png` 作为基准图（若不存在则生成模拟图像），并使用预设的若干模板位置。

    出口:
        无返回值。函数在控制台打印每次实验结果与误差，并展示若干可视化图（3D 曲面、2D 热力图、局部放大）。

    说明:
        - 返回的结果通过全局打印与窗口可视化呈现，便于分析不同方法的定位精度。
    """
    print("\n" + "="*50)
    print("主实验：子图定位对比分析")
    print("="*50)
    
    # 读取基准图像
    try:
        base_img = cv2.imread('pic1.png', cv2.IMREAD_GRAYSCALE)
        if base_img is None:
            raise FileNotFoundError
    except:
        base_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        base_img[50:150, 50:150] = 255
        base_img[100:120, 100:120] = 0
    
    base_img = base_img.astype(np.float64)
    
    # 6个子图的真实位置 [x, y, 宽度, 高度]
    true_positions = [
        [50, 200, 100, 100],
        [80, 80, 60, 60],
        [120, 30, 80, 80],
        [30, 100, 90, 90],
        [150, 150, 70, 70],
        [10, 10, 50, 50]
    ]
    
    results_spatial = []
    results_freq = []
    results_phase_only = []
    
    print("\n{:<10} {:<20} {:<20} {:<15}".format(
        "实验次数", "真实位置 (x,y)", "空域结果 (x,y)", "空域误差 (Δx,Δy)"))
    print("-"*65)
    
    for idx, pos in enumerate(true_positions, 1):
        x, y, w, h = pos
        template = base_img[y:y+h, x:x+w].copy()
        
        # 三种方法计算
        rho_spatial, peak_spatial = spatial_correlation(base_img, template)
        rho_freq, peak_freq = frequency_correlation(base_img, template)
        rho_phase, peak_phase = phase_only_correlation(base_img, template)
        
        # 计算误差
        error_spatial = (abs(peak_spatial[1] - x), abs(peak_spatial[0] - y))
        error_freq = (abs(peak_freq[1] - x), abs(peak_freq[0] - y))
        error_phase = (abs(peak_phase[1] - x), abs(peak_phase[0] - y))
        
        results_spatial.append((peak_spatial, error_spatial))
        results_freq.append((peak_freq, error_freq))
        results_phase_only.append((peak_phase, error_phase))
        
        print("{:<10} ({:<3},{:<3}) {:<15} ({:<3},{:<3}) {:<15} ({},{})".format(
            f"第{idx}次", x, y, "", peak_spatial[1], peak_spatial[0], "", error_spatial[0], error_spatial[1]))
        
        # ===== 3D mesh图可视化=====
        if idx <= 6:
            fig = plt.figure(figsize=(20, 8))
            
            # ---------- 空域相关 ----------
            ax1 = fig.add_subplot(131, projection='3d')
            X1 = np.arange(0, rho_spatial.shape[1])
            Y1 = np.arange(0, rho_spatial.shape[0])
            X1, Y1 = np.meshgrid(X1, Y1)
            surf1 = ax1.plot_surface(X1, Y1, rho_spatial, cmap='viridis', 
                                     alpha=0.7, rstride=5, cstride=5)
            peak_z_spatial = rho_spatial[peak_spatial[0], peak_spatial[1]]
            ax1.scatter(peak_spatial[1], peak_spatial[0], peak_z_spatial, 
                       color='red', s=300, marker='*', linewidths=2, edgecolors='darkred',
                       label=f'峰值({peak_spatial[1]},{peak_spatial[0]})', 
                       zorder=10, depthshade=False)
            z_range_spatial = rho_spatial.max() - rho_spatial.min()
            ax1.set_zlim(rho_spatial.min() - z_range_spatial * 0.1, 
                        rho_spatial.max() + z_range_spatial * 0.3)
            
            ax1.set_title(f'空域相关曲面\n峰值=({peak_spatial[1]},{peak_spatial[0]}) 真实=({x},{y})\n'
                         f'误差: Δx={error_spatial[0]}, Δy={error_spatial[1]}\n'
                         f'峰值高度: {peak_z_spatial:.4f}', 
                         fontsize=10, fontweight='bold')
            ax1.set_xlabel('X坐标', fontsize=9)
            ax1.set_ylabel('Y坐标', fontsize=9)
            ax1.set_zlabel('相关系数', fontsize=9, color='blue')
            
            textstr_spatial = f'子图位置: ({x}, {y})\n峰值: {peak_z_spatial:.4f}\n误差: ({error_spatial[0]}, {error_spatial[1]})'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text2D(0.02, 0.98, textstr_spatial, transform=ax1.transAxes, 
                      fontsize=9, verticalalignment='top', bbox=props)
            
            ax1.legend(loc='upper right', fontsize=8)
            
            # ---------- 频域相关 ----------
            ax2 = fig.add_subplot(132, projection='3d')
            X2 = np.arange(0, rho_freq.shape[1])
            Y2 = np.arange(0, rho_freq.shape[0])
            X2, Y2 = np.meshgrid(X2, Y2)
            surf2 = ax2.plot_surface(X2, Y2, rho_freq, cmap='plasma', 
                                    alpha=0.7, rstride=5, cstride=5)
            peak_z_freq = rho_freq[peak_freq[0], peak_freq[1]]
            ax2.scatter(peak_freq[1], peak_freq[0], peak_z_freq, 
                       color='red', s=300, marker='*', linewidths=2, edgecolors='darkred',
                       label=f'峰值({peak_freq[1]},{peak_freq[0]})', 
                       zorder=10, depthshade=False)
            z_range_freq = rho_freq.max() - rho_freq.min()
            ax2.set_zlim(rho_freq.min() - z_range_freq * 0.1, 
                        rho_freq.max() + z_range_freq * 0.3)
            
            ax2.set_title(f'频域全频谱相关曲面\n峰值=({peak_freq[1]},{peak_freq[0]}) 真实=({x},{y})\n'
                         f'误差: Δx={error_freq[0]}, Δy={error_freq[1]}\n'
                         f'峰值高度: {peak_z_freq:.2e}', 
                         fontsize=10, fontweight='bold', color='darkred')
            ax2.set_xlabel('X坐标', fontsize=9)
            ax2.set_ylabel('Y坐标', fontsize=9)
            ax2.set_zlabel('相关系数', fontsize=9, color='purple')
            
            textstr_freq = f'子图位置: ({x}, {y})\n峰值: {peak_z_freq:.2e}\n误差: ({error_freq[0]}, {error_freq[1]})'
            props_freq = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
            ax2.text2D(0.02, 0.98, textstr_freq, transform=ax2.transAxes, 
                      fontsize=9, verticalalignment='top', bbox=props_freq)
            
            ax2.legend(loc='upper right', fontsize=8)
            
            # ---------- 相位谱相关 ----------
            ax3 = fig.add_subplot(133, projection='3d')
            X3 = np.arange(0, rho_phase.shape[1])
            Y3 = np.arange(0, rho_phase.shape[0])
            X3, Y3 = np.meshgrid(X3, Y3)
            surf3 = ax3.plot_surface(X3, Y3, rho_phase, cmap='coolwarm', 
                                    alpha=0.8, rstride=5, cstride=5)
            peak_z_phase = rho_phase[peak_phase[0], peak_phase[1]]
            ax3.scatter(peak_phase[1], peak_phase[0], peak_z_phase, 
                       color='red', s=350, marker='*', linewidths=2, edgecolors='darkred',
                       label=f'峰值({peak_phase[1]},{peak_phase[0]})', 
                       zorder=10, depthshade=False)
            z_range_phase = rho_phase.max() - rho_phase.min()
            if z_range_phase < 1e-6:
                ax3.set_zlim(rho_phase.min() - 1e-6, rho_phase.max() + 5e-4)
            elif z_range_phase < 1e-3:
                ax3.set_zlim(rho_phase.min() - z_range_phase * 0.2, 
                            rho_phase.max() + z_range_phase * 1.0)
            else:
                ax3.set_zlim(rho_phase.min() - z_range_phase * 0.1, 
                            rho_phase.max() + z_range_phase * 0.3)

            ax3.set_title(f'相位谱相关曲面\n峰值=({peak_phase[1]},{peak_phase[0]}) 真实=({x},{y})\n'
                         f'误差: Δx={error_phase[0]}, Δy={error_phase[1]}\n'
                         f'峰值高度: {peak_z_phase:.2e}', 
                         fontsize=10, fontweight='bold', color='darkgreen')
            ax3.set_xlabel('X坐标', fontsize=9)
            ax3.set_ylabel('Y坐标', fontsize=9)
            ax3.set_zlabel('相关系数', fontsize=9, color='darkred')
            
            textstr_phase = f'子图位置: ({x}, {y})\n峰值: {peak_z_phase:.2e}\n误差: ({error_phase[0]}, {error_phase[1]})'
            props_phase = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
            ax3.text2D(0.02, 0.98, textstr_phase, transform=ax3.transAxes, 
                      fontsize=9, verticalalignment='top', bbox=props_phase)
            
            ax3.legend(loc='upper right', fontsize=8)
            
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
            fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

            fig.suptitle(f'第{idx}次实验：子图定位对比分析  (子图位置: x={x}, y={y})', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
    
    # 结果表格
    print("\n表1：空域实验数据表")
    print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}".format(
        "实验次数", "计算结果x", "计算结果y", "真实位置x", "真实位置y", "Δx", "Δy"))
    print("-"*70)
    for i, ((peak, error), true_pos) in enumerate(zip(results_spatial, true_positions), 1):
        print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}".format(
            i, peak[1], peak[0], true_pos[0], true_pos[1], error[0], error[1]))
    
    print("\n表2：频域实验数据表")
    print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}".format(
        "实验次数", "计算结果x", "计算结果y", "真实位置x", "真实位置y", "Δx", "Δy"))
    print("-"*70)
    for i, ((peak, error), true_pos) in enumerate(zip(results_freq, true_positions), 1):
        print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}".format(
            i, peak[1], peak[0], true_pos[0], true_pos[1], error[0], error[1]))
    
    print("\n表3：频域(仅相位)实验数据表")
    print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}".format(
        "实验次数", "计算结果x", "计算结果y", "真实位置x", "真实位置y", "Δx", "Δy"))
    print("-"*70)
    for i, ((peak, error), true_pos) in enumerate(zip(results_phase_only, true_positions), 1):
        print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}".format(
            i, peak[1], peak[0], true_pos[0], true_pos[1], error[0], error[1]))

    # 实验结论
    print("\n" + "="*50)
    print("实验结论")
    print("="*50)
    
    spatial_errors = np.array([err for _, err in results_spatial])
    freq_errors = np.array([err for _, err in results_freq])
    phase_errors = np.array([err for _, err in results_phase_only]) # 添加
    
    print(f"\n1. 平均定位误差：")
    print(f"   空域方法 (NCC)：  Δx={np.mean(spatial_errors[:,0]):.2f}, Δy={np.mean(spatial_errors[:,1]):.2f}")
    print(f"   频域(全频谱)： Δx={np.mean(freq_errors[:,0]):.2f}, Δy={np.mean(freq_errors[:,1]):.2f}")
    print(f"   频域(仅相位)： Δx={np.mean(phase_errors[:,0]):.2f}, Δy={np.mean(phase_errors[:,1]):.2f}")

# ==================== 6. 运行主程序 ====================
if __name__ == "__main__":
    I1, I3 = phase_spectrum_experiment()
    main_experiment()
    print("\n" + "="*50)
    print("所有实验完成！")
    print("="*50)