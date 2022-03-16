from pyexpat import model
from model import *
from data import *

model = load_model("D:/Python/Python project/TestModel/unet_n.hdf5")

testGene = testGenerator("D:/Python/Python project/TestModel/unet_test _mask/data_test/membrane/test")

results = model.predict(testGene,30,verbose=1)
#为来自数据生成器的输入样本生成预测
#30是step,steps: 在停止之前，来自 generator 的总步数 (样本批次)。 可选参数 Sequence：如果未指定，将使用len(generator) 作为步数。
#上面的返回值是：预测值的 Numpy 数组。
#steps:在声明了一个epoch完成，并开始下一个epoch之前从生成器产生的总步数
#works：最大进程数量
#use_multiprocessing:多线程

saveResult("D:/Python/Python project/TestModel/unet_test _mask/data_test/membrane/test",results)
