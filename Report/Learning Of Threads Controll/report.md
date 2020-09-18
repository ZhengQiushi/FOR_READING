# 阅读外校代码
## 北理
### 分3个线程：
- 生成图像（摄像头读取图像存入Mat）
- 处理图像
- 与电控通信

![北理](北理.png)

## 上交
### 分2个线程：
- 接受电控的信息
- 主线程、向电控发送信息

## 太原理工
### 分2个线程：
- 接受电控的信息
- 主线程、向电控发送信息(坐标转换时使用mutex保护数据？)

# 思考
1. 是否需要面向线程重构当前代码
2. 是否考虑删减当前代码中对于线程的使用