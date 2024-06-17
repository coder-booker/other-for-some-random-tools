import os
import shutil
import numpy as np

def split_data_set(src_dir, dst_dir, ratio):
    # 获取所有的文件名
    file_names = [f[:-4] for f in os.listdir(os.path.join(src_dir, 'images')) if f.endswith('.jpg')]

    # 计算测试集的数量
    test_num = int(len(file_names) * ratio)

    # 随机抽取测试集
    test_set = np.random.choice(file_names, test_num, replace=False)
    train_set = list(set(file_names) - set(test_set))

    # 创建目标目录
    os.makedirs(os.path.join(dst_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'test', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'train', 'labels'), exist_ok=True)

    # 复制文件
    for name in test_set:
        shutil.copy(os.path.join(src_dir, 'images', name + '.jpg'), os.path.join(dst_dir, 'test', 'images', name + '.jpg'))
        shutil.copy(os.path.join(src_dir, 'labels', name + '.txt'), os.path.join(dst_dir, 'test', 'labels', name + '.txt'))

    for name in train_set:
        shutil.copy(os.path.join(src_dir, 'images', name + '.jpg'), os.path.join(dst_dir, 'train', 'images', name + '.jpg'))
        shutil.copy(os.path.join(src_dir, 'labels', name + '.txt'), os.path.join(dst_dir, 'train', 'labels', name + '.txt'))
abs_src = r'D:\learning\github_clone_repo\other-for-some-random-tools\dataset\car_dataset'
abs_dst = r'D:\learning\github_clone_repo\other-for-some-random-tools\dataset\car_dataset_processed'
test_set_ratio = 0.2
split_data_set(abs_src, abs_dst, test_set_ratio)