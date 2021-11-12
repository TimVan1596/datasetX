import os
import numpy as np
from PIL import Image
from test.reg_utils import my_predict
from shutil import copy
from sys import exit


# 主驱动
# parameters是训练出来的参数
def pic_to_np(path, label_dict, pic_size: 224):
    parameter_path = dtx_config.get('parameter_path')
    parameters = load_parameter(parameter_path)

    # 根据label下的标签提前创建文件夹
    for label in label_dict.values():
        create_folder(path + '/' + label)

    pic_list = os.listdir(path)
    for pic in pic_list:
        pic_path = path + '/' + pic
        # 排除文件夹
        if os.path.isfile(pic_path):
            # 图片经过检测后将返回一个int型的结果
            result = detect_pic(pic_path=pic_path, parameters=parameters, pic_size=pic_size)
            label = label_dict.get(result)
            # 目的地的地址
            dst_path = path + '/' + label+'/'
            my_copyfile(pic_path, dst_path)


# 负责对图像进行处理和检测
def detect_pic(pic_path, parameters, pic_size):
    # 缩放
    image = Image.open(pic_path)
    re_image = image.resize((pic_size, pic_size))
    # 转换成numpy数组(仅一行)
    image_arr = np.array(re_image)
    # reshape的参数为-1代表自动计算剩余
    train_x_flatten = image_arr.reshape(-1, 1)
    result = int(my_predict(train_x_flatten, parameters).squeeze())
    assert type(result) == int
    return result


# 将图片保存到指定path下的label标签
# label = 要存入的标签
def my_copyfile(src_file, dst_path):  # 复制函数
    if not os.path.isfile(src_file):
        print("%s not exist!" % src_file)
    else:
        file_path, file_name = os.path.split(src_file)  # 分离文件名和路径
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)  # 创建路径
        copy(src_file, dst_path + file_name)  # 复制文件


# 负责加载parameter文件
def load_parameter(parameter_path):
    d2 = np.load(parameter_path, allow_pickle=True)
    return d2.item()


# 判断文件夹是否存在，不存在则创建
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    # 自动将图片分类的参数
    # 注意：label_dict里是数字是key,标签是value
    dtx_config = {
        'path': './origin/'
        # 标签字典
        , 'label_dict': {
            1: 'yes',
            0: 'no'
        }
        # 训练集占的比例
        , 'train_percent': 0.9
        # 图片缩放的尺寸
        , 'pic_size': 224
        , 'isLog': True
        # 训练出来的模型结果
        , 'parameter_path': './test/parameters.npy'
    }

    pic_to_np(path=dtx_config.get('path')
              , label_dict=dtx_config.get('label_dict')
              , pic_size=dtx_config.get('pic_size'))
    pass
