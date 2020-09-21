#include<atomic>
#include<mutex>
/* INFO：	1.重构时根据实际需求增加 mutex、atomic以及shared data（需注意数据保护）
			2.可根据需求增加线程，例如：图像保存线程、将电控通信线程拆分成接收与发送等
*/



class ThreadControl{
private:
	struct __FrameSt	//与电控通信的数据
	{
		/*头帧 时间戳 yaw pitch 标识符 校验 尾帧 */
		uint8_t head = 0xf1;
		uint16_t timeStamp = 0;
		float yaw = 0.0;
		float pitch = 0.0;
		uint8_t extra[2] = {0, 0}; // additional imformation
		uint8_t crc8check = 0;
		uint8_t end = 0xf2;
	} m_frame;

	std::atomic<bool> enemy_is_red;	// 功能：表示敌人车辆颜色
										// 使用：在初始化时确定其数值

    std::atomic<bool> ready_to_receive;	// 功能：已准备接收电控数据
										// 使用：每次通信前后改写其数值

	std::atomic<bool> ready_to_send;	// 功能：已准备发送给电控数据			
										// 使用：每次通信前后改写其数值

	std::atomic<int> m_WorkMode;		// 功能：显示当前工作模式:——摸鱼, 自瞄, 大风车
										// 使用：从电控读取工作模式时改写其数值
	
	std::mutex m_frame_ready;			// 功能：确保向m_frame写入数据的过程不会被打断
									// 使用：向m_frame写入数据时上锁，写入结束后解锁

	

public:
	ThreadControl();
	~ThreadControl();

	/* 线程作用： 自瞄主循环，解算角度 */
	void AutoAim();

	/* 线程作用： 与电控通信 */
	void Communicate();

	
};