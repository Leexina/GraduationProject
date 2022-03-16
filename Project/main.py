from model import *
from data import *
#导入这两个文件中的所有函

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#数据增强时变换方式的字典
data_gen_args = dict(rotation_range=0.2,        #旋转
                    width_shift_range=0.05,     #宽度变化
                    height_shift_range=0.05,    #高度变化
                    shear_range=0.05,   #错切变化
                    zoom_range=0.05,    #缩放
                    horizontal_flip=True,   #水平翻转
                    fill_mode='nearest')    #填充模式

myGene = trainGenerator(2,'D:/Python/Python project/TestModel/unet_test _mask/data_test/membrane/train',
                        'image','label',data_gen_args , save_to_dir = None)
#得到一个生成器，以batch_size=2的速率无限生成增强后的数据

model = unet()
model_checkpoint = ModelCheckpoint('unet_high.hdf5', monitor='loss',verbose=1, save_best_only=True)
#回调函数，在每个epoch后保存模型到filepath
#filename:保存模型的路径
#mointor:需要监视的值检测Loss使它最小
#verbose:信息展示模式，1展示，0不展示
#save_best_only:保存在验证集上性能最好的模型

model.fit(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])
#训练函数
#generator:生成器（image,mask)
#steps_per_epoch:训练steps_per_epoch个数据时记一个epoch结束
#step_per_epoch指的是每个epoch有多少个batch_size，也就是训练集样本总数除以batch_size的值
#epoch：数据迭代轮数
#callbacks：回调函数

testGene = testGenerator("D:/Python/Python project/TestModel/unet_test _mask/data_test/membrane/test")

results = model.predict(testGene,30,verbose=1)
#为来自数据生成器的输入样本生成预测
#30是step,steps: 在停止之前，来自 generator 的总步数 (样本批次)。 可选参数 Sequence：如果未指定，将使用len(generator) 作为步数。
#上面的返回值是：预测值的 Numpy 数组。
#steps:在声明了一个epoch完成，并开始下一个epoch之前从生成器产生的总步数
#works：最大进程数量
#use_multiprocessing:多线程

saveResult("D:/Python/Python project/TestModel/unet_test _mask/data_test/membrane/test",results)
#保存结果