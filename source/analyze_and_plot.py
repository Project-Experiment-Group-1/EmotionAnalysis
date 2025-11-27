import numpy as np
import matplotlib.pyplot as plt
import train_emotions as emotion_lib  # 导入你刚才修改好的库文件

# ==========================================
# 1. 准备工作
# ==========================================
print("正在加载数据...")
try:
    # 调用库函数加载并切分数据
    # 注意：如果你的 data 文件夹不在上一级目录，请修改 path_data
    data = emotion_lib.load_and_split_data(full_features=False, path_data='../data/')
except FileNotFoundError as e:
    print(f"错误: {e}")
    exit()

# 从字典中解包出训练集和验证集
X_train, y_train = data['train']
X_val, y_val = data['val']

# ==========================================
# 2. 循环分析 (核心逻辑)
# ==========================================
max_components = 50  # 我们想测试的最大组件数
results = []  # 用来存跑出来的结果

print(f"开始分析 1 到 {max_components} 个组件的效果...")

for n in range(1, max_components + 1):
    # A. 训练模型 (调用库函数)
    # 就像搭积木一样，只需要告诉它用多少个组件(n)
    model = emotion_lib.train_model(X_train, y_train, n_components=n)

    # B. 验证模型
    y_pred_val = model.predict(X_val)

    # C. 计算指标 (调用库函数)
    # mse 是一个包含3个数字的列表 [Arousal_MSE, Valence_MSE, Intensity_MSE]
    mse = emotion_lib.get_mse(y_val, y_pred_val)
    # ccc 也是一个包含3个数字的列表
    ccc = emotion_lib.get_ccc(y_val, y_pred_val)

    # D. 记录结果
    # 我们把这次循环的所有数据打包存起来
    results.append({
        'n_components': n,
        'mse': mse,  # [MSE_A, MSE_V, MSE_I]
        'ccc': ccc,  # [CCC_A, CCC_V, CCC_I]
        'avg_mse': np.mean(mse),
        'avg_ccc': np.mean(ccc)
    })

    # 每5次打印一下进度，避免在这里干等
    if n % 5 == 0:
        print(f"已完成: {n}/{max_components} (当前平均 CCC: {np.mean(ccc):.4f})")

for r in results:
    print('Training PLS with ' + str(r['n_components']) + ' components. MSE=' + str(r['mse'][1]) + ' CCC=' + str(r['ccc'][1]))

# ==========================================
# 3. 绘制图表
# ==========================================
print("正在绘图...")

# 提取绘图需要的数据列
ns = [r['n_components'] for r in results]

# 提取 Arousal, Valence, Intensity 的具体指标
# r['mse'][0] 代表 Arousal, [1] 代表 Valence, [2] 代表 Intensity
mse_arousal = [r['mse'][0] for r in results]
mse_valence = [r['mse'][1] for r in results]
mse_intensity = [r['mse'][2] for r in results]

ccc_arousal = [r['ccc'][0] for r in results]
ccc_valence = [r['ccc'][1] for r in results]
ccc_intensity = [r['ccc'][2] for r in results]

# 创建一个宽一点的画布，放两张图
plt.figure(figsize=(15, 6))

# --- 左图：MSE (越低越好) ---
plt.subplot(1, 2, 1)
plt.plot(ns, mse_arousal, label='Arousal', marker='.', alpha=0.7)
plt.plot(ns, mse_valence, label='Valence', marker='.', alpha=0.7)
plt.plot(ns, mse_intensity, label='Intensity', marker='.', alpha=0.7)
plt.xlabel('Number of Components')
plt.ylabel('MSE')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- 右图：CCC (越高越好) ---
plt.subplot(1, 2, 2)
plt.plot(ns, ccc_arousal, label='Arousal', marker='.', alpha=0.7)
plt.plot(ns, ccc_valence, label='Valence', marker='.', alpha=0.7)
plt.plot(ns, ccc_intensity, label='Intensity', marker='.', alpha=0.7)
plt.title('Validation CCC (Higher is Better)')
plt.xlabel('Number of Components')
plt.ylabel('Concordance Correlation Coefficient')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 4. 输出结论
# ==========================================
# 找出平均 CCC 最高的那个配置
best_run = max(results, key=lambda x: x['avg_ccc'])
print("-" * 30)
print("分析完成！建议结果：")
print(f"最佳组件数量: {best_run['n_components']}")
print(f"最佳平均 CCC: {best_run['avg_ccc']:.4f}")
print("-" * 30)