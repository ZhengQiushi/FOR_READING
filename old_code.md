###老代码阅读


###main
1、线程数初始化
2、通信选择USB通信/串口通信
3、开视频或摄像头
4、开图像显示辅助程序
5、attack线程组：
1）各线程对attack、windmill初始化
2）风车实例化
TensorFlow创建session,从pb文件读取模板，并将模板设置到session里
/*TensorFlow
图容器中包括：
张量(tensor)TensorFlow使用tensor数据结构代表所有的数据，可看作n维数组或列表。
变量(Variable):常用于定义模型中的参数，是通过不断训练得到的值。
占位符(placeholder):输入变量的载体。也可以理解成定义函数时的参数。
节点操作(op):一个op获得0或多个的Tensor，执行计算，产生0或多个Tensor，是张量中的运算关系。

一个Tensorflow图必须在会话(session)里启动会话将图的op分法到COU或GPU的设备上，同时执行op的方法，返回产生的tensor.

session与图的交互过程中定义了两种数据流向机制：
注入机制(feed):通过占位符向模式传入数据
取回机制(fetch):从模式中取得结果
*/
3）wait_and_get阻塞获得的图像，保证线程安全
frame图像
timeStamp时间戳：标记时序即每一帧编号
func获得图像时须调用的函数
4)getWorkMode获得工作模式
模式前缀RM_WINDMILL指定0线程跑风车，并根据后面具体模式选择风车工作模式
RM_AUTO_ATTACK模式跑自瞄

### attack

1、初始化、并根据历史的尺寸、相对帧编号判断是否启用ROI

2、m_preDetect预检测
获得装甲板（膨胀操作）

3、分类器：使用基于tensorflow的分类器

4、根据时间戳判断：如果已经有更新的一帧发出去了，返回false

5、目标匹配：之前未选择过目标/选择过，找一样的

6、预测采用卡尔曼滤波的算法
算法思想：根据当前仪器"测量值"和上一刻"预测量"和"误差",计算得到当前的最优量再预测下一刻的量

7、PID算法对yaw的修正
PID控制器根据系统的误差，利用比例、积分、微分计算出控制量进行控制。

8、结果发给电控

###windmill

过程与自瞄相似，不同之处是可能受图形形状等影响比较好识别没有使用分类器，切用了solvepnp来计算paw、pitch、x、y、z坐标

###imageshow

###ImageShowClient 子线程绘图客户类
主要功能：
1、
m_clockPrint()：输出 clock()计时的结果, 使用互斥锁, 保证 client 间输出不互相打断
//在多线程环境中，多个线程竞争一个公共资源时，容易引发线程安全问题，因此需要引入锁的机制，保证人以时刻只有一个线程在访问公共资源
clock():对name进行配对，多次调用去最远两次，计算之间的耗时

2、update(const cv::Mat &frame, int frameTick)
   更新图像
3、绘制各种图像(轮廓、灯条、目标等)

4、show()
尝试将图像发给服务端，如果服务端繁忙就不发送。用exchange记录编号，服务端通过编号读图像

###ImageShowServer 主线程图像显示服务类
mainloop()处理键鼠操作、显示ImageShowClient传来的图像(s_currentCallID为编号)

###communicator
1、获取pitch、yaw、工作模式，将yaw和pitch进入双端队列

2、crc校验：循环冗余校检查是一种数据传输检错功能，对数据进行多项式计算，并将得到的结果附在帧的后面，接收设备也执行类似的算法，保证数据传输的正确性和完整性

###capture
1、各摄像头打开、初始化等操作

2、play()采集图像

3、setCaptureROI()开启ROI

4、wait_and_get()阻塞获得图像，保证线程安全






























