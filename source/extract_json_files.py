import os
import shutil


def extract_json_files(src_root, dst_root):
    """
    保留目录结构，只提取JSON文件到新目录。
    新目录结构将为: dst_root / 源文件夹名 / 子文件夹
    """
    # 1. 检查源目录是否存在
    if not os.path.exists(src_root):
        print(f"错误：源目录 '{src_root}' 不存在。")
        return

    # 获取源文件夹的名称（处理路径末尾可能的斜杠）
    src_folder_name = os.path.basename(os.path.normpath(src_root))

    # 2. 统计计数器
    folder_count = 0
    file_count = 0

    print(f"正在开始处理...")
    print(f"源目录: {src_root}")
    # 这里的最终存储位置实际上变成了 dst_root/src_folder_name
    final_dst_path = os.path.join(dst_root, src_folder_name)
    print(f"提取位置: {final_dst_path}")
    print("-" * 30)

    # 3. 遍历源目录中的所有内容
    # os.walk 会递归遍历所有子文件夹
    for root, dirs, files in os.walk(src_root):

        # 计算当前目录相对于源根目录的相对路径
        # 例如：从 "C:/Data/001" 变成 "001"
        relative_path = os.path.relpath(root, src_root)

        # 构建目标目录的完整路径
        # 修改点：在 dst_root 和 relative_path 之间插入了 src_folder_name
        target_dir_path = os.path.join(dst_root, src_folder_name, relative_path)

        # 4. 筛选：只处理包含 json 文件的文件夹
        # (虽然也会创建空文件夹，但这样能完美保留结构)
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
            folder_count += 1

        for filename in files:
            # === 核心逻辑：忽略 JPG，只选 JSON ===
            if filename.lower().endswith('.json'):
                src_file = os.path.join(root, filename)
                dst_file = os.path.join(target_dir_path, filename)

                # 复制文件 (copy2 会保留文件的时间戳等元数据)
                shutil.copy2(src_file, dst_file)
                file_count += 1
                print(f"已提取: {os.path.join(src_folder_name, relative_path, filename)}")

    print("-" * 30)
    print(f"处理完成！")
    print(f"共创建文件夹: {folder_count} 个")
    print(f"共提取 JSON: {file_count} 个")
    print(f"所有文件保存在: {os.path.join(dst_root, src_folder_name)}")


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 请将下面的路径修改为你实际的文件夹路径
    # 建议使用绝对路径，注意 Windows 路径要用 r'' 或双斜杠 //

    # 你的原始文件夹（例如叫 MySourceData）
    SOURCE_FOLDER = r"D:\xinshengcheng\AFEW-VA"

    # 结果存放的根目录。
    # 脚本运行后，会在这个文件夹下自动创建一个叫 "SourceData" 的文件夹
    TARGET_FOLDER = r"C:\Projects\facial-expression-analysis-main\data"
    # ===========================================

    extract_json_files(SOURCE_FOLDER, TARGET_FOLDER)