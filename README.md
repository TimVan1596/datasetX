# datasexX
#### 介绍
根据您提供的图片集合，自动生成格式为.h5的数据集。并可以进行训练和分类。

#### 使用说明
简洁版：先运行gen_dataset.py ,再运行./test/test_datasetx.py，最后运行auto_file.py

1.当前目录应该包含data文件夹，其下应该有标签名的文件夹，标签名文件夹中是对应的图片
1.  gen_dataset.py ，配置dtx_config的参数，默认在./output/下产生训练集、测试集、配置等文件
2.  ./test/test_datasetx.py，对数据集进行训练，并在./test/文件夹下保存训练出的参数parameter
3.  auto_file.py，读取训练出的参数parameter对origin文件夹下的图片进行分类（假设origin里是您准备进行分类的图片）



#### 软件架构
python



