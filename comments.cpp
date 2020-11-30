 #include "attack.hpp"
#include <windmill/Windmill.hpp>
#include "capture.hpp"
#include "communicator.hpp"
#include "imageshow.hpp"

int main()
{
    std::cout << "Using OpenCV " << CV_VERSION << std::endl;
    int threadNum = armor::stConfig.get<int>("auto.thread-num");//线程数初始化

    /* 通信 */
#ifdef USE_USB
    armor::CommunicatorUSB communicator;
    communicator.open(0x0477, 0x5620);
#else
    /*
    设置串口通信:
    打开设备 open()(含守护线程和转移时间片）
    信息发送 send()   
    读值校验更新 startReceiveService()（开始接收线程）
    */
    armor::CommunicatorSerial communicator;
    communicator.disable(!armor::stConfig.get<bool>("communicator.enable"));
    communicator.open(armor::stConfig.get<std::string>("communicator.serial-port"));
#endif
    communicator.startReceiveService(); //开始接收线程

    /*摄像头开启
      init()初始化
      play()开始传输图像；
      setCaotureROI()开启ROI
      wait_and_get()阻塞以获取摄像头
      getCurrentInterval()返回采集时间间隔
    */
    
    armor::Capture* cap = nullptr;
    armor::LajiVision lajiVision;
    armor::DaHuaVision dahuaVision;

    if (armor::stConfig.get<bool>("cap.is-video")) {//打开视频或者垃圾摄像头
        cap = &lajiVision;
    }
    else {
        cap = &dahuaVision;//打开大华摄像头
    }

    cap->init(); // 初始化

    if (armor::stConfig.get<bool>("cap.is-video")) {
        cap->setCaptureROI(cv::Size2i(1280, 900), armor::CAP_ROI_CENTER_CENTER); // 采集ROI
    }
    else {
        cap->setCaptureROI(cv::Size2i(1280, 900), armor::CAP_ROI_CENTER_CENTER); // 采集ROI
    }
    cap->play(); //开始采集图像

    //include imageshow.hpp  
    /* 开图像显示辅助程序
       setMode()配置模式
       setFontSize()设置字体
       enableClockPrint()输出clock计时
       enableAverageCostPrint()输出CPU平均耗时
     
    */
    armor::ImageShowServer isServer(threadNum, 0.5);//构造
   //配置模式，三种，具体看配置文件
    isServer.setMode(armor::stConfig.get<int>("isServer.mode"));//配置模式，三种，具体看配置文件
    isServer.setFontSize(1.25);
    isServer.enableClockPrint(true);
    isServer.enableAverageCostPrint(true);

    /* attack线程组 */
    std::vector <std::thread> attackThreads;
    attackThreads.resize(threadNum);
    armor::PID pid;//电控的？好像不用管
    pid.init(armor::stConfig.get<double>("auto.kp"),
        armor::stConfig.get<double>("auto.ki"),
        armor::stConfig.get<double>("auto.kd"));

    for (int i = 0; i < threadNum; ++i) {//首先每个线程都会对attack、windmill进行初始化，之后会根据循环中i的值来为不同到线程分配不同到击打任务（自瞄或者风车击打）
        attackThreads[i] = std::thread([cap, &isServer, i, &communicator, &pid]() {
            armor::ImageShowClient isClient = isServer.getClient(i);//include imageshow.hpp

        /* 初始化 attack 
               mode为RM_AUTO_ATTACK时执行自瞄
               开始自瞄..自瞄里面的代码有些没看懂
               首先清除旧数据
               m_preDetect--预处理获得装甲板（hsv，膨胀，面积比，长宽比）
               mat2tensor--预处理好的Mat 转化为tensor类
               getThreshold--得到二值化阈值
               init_my_tf看不懂
               m_classify_single_tensor--分类器
               m_match--策略--选好的装甲板如何击
               getBoundingRect--图像扩展ROI，改变画面大小
               run：
                   __处理多线程新旧数据是要干撒 
                   __m_preDetect
                   __m_classsify_single_tensor
                   __m_preTargets存放预处理结果
                   __目标匹配+预测+修正弹道+计算欧拉角+射击策略
         */

            armor::Attack attack(communicator, pid, isClient);
            attack.enablePredict(armor::stConfig.get<bool>("auto.enable-predict"));//是否进行预测

            attack.setMode(armor::stConfig.get<std::string>("attack.attack-color") == "red");//击打颜色为红色，具体见配置文件

            /* 初始化 windmill 
               mode为RM_WINDMILL时执行风车
               清楚旧数据
               预处理
               筛选
               pnp算角度以及坐标
               实例化中0线程跑风车
            */
            cv::Mat TvCtoL = (cv::Mat_<double>(3, 1) << armor::stConfig.get<double>("power.x"), armor::stConfig.get<double>("power.y"), armor::stConfig.get<double>("power.z"));  //摄像头到云台转化矩阵


            double delay = armor::stConfig.get<double>("power.delay");
            double maxPitchError = armor::stConfig.get<double>("power.maxPitchError");
            double maxYawError = armor::stConfig.get<double>("power.maxYawError");


            /* 风车实例化 
            */
            wm::Windmill* pWindMill =
                wm::Windmill::GetInstance(armor::stCamera.camMat, armor::stCamera.distCoeffs, TvCtoL, delay, "../rmdl/models/symbol.onnx", &isClient, maxPitchError, maxYawError);

            int64_t timeStamp = 0;
            cv::Mat frame;

            cap->initFrameMat(frame);
            float gYaw = 0.0;
            float gPitch = 0.0;

            while ((cap->isOpened()) && !isServer.isWillExit()) {


                if (cap->wait_and_get(frame, timeStamp, [&communicator, &gYaw, &gPitch]() {

                    //                    communicator.getGlobalAngle(&gYaw, &gPitch);
                    })) {

                    /* 刷新主线程窗口图像 */
                    isClient.update(frame, int(timeStamp / 1000));
                    isClient.addText(cv::format("ts %lld", timeStamp));
                    isClient.addText(cv::format("1/fps %2.2f ms", cap->getCurrentInterval() / 1000.0));
                    isClient.addText(cv::format("send %2.2f ms", communicator.getCurrentInterval() / 1000.0));

                    isClient.clock("run");
                    //                    communicator.getGlobalAngle(&gYaw, &gPitch);
                                        /* core code */
                    auto mode = communicator.getWorkMode();
                    //auto mode =  armor::RM_WINDMILL_SMALL_CLOCK;
                    isClient.addText(cv::format("mode: %x", int(mode)));

                    switch (mode) {
                    case armor::RM_WINDMILL_SMALL_CLOCK:
                    case armor::RM_WINDMILL_SMALL_ANTIC:
                    case armor::RM_WINDMILL_LARGE_CLOCK:
                    case armor::RM_WINDMILL_LARGE_ANTIC:
                        // 指定运行线程
                        if (i == 0) {//i=0的线程跑风车
                            /* 大风车 */
                            float pitch = 0.0;
                            float yaw = 0.0;
                            switch (mode) {//选择模式初始化参数（这是旧版，最新版未更新）
                            case armor::RM_WINDMILL_SMALL_CLOCK:
                                isClient.addText("RM_WINDMILL_SMALL_CLOCK");
                                pWindMill->delay = delay;
                                pWindMill->maxPitchError = maxPitchError;
                                pWindMill->maxYawError = maxYawError;
                                pWindMill->mode = "linear";
                                break;
                            case armor::RM_WINDMILL_SMALL_ANTIC:
                                isClient.addText("RM_WINDMILL_SMALL_ANTIC");
                                pWindMill->delay = -delay;
                                pWindMill->maxPitchError = -maxPitchError;
                                pWindMill->maxYawError = -maxYawError;
                                pWindMill->mode = "linear";
                                break;
                            case armor::RM_WINDMILL_LARGE_CLOCK:
                                isClient.addText("RM_WINDMILL_CLOCK");
                                pWindMill->maxPitchError = maxPitchError;
                                pWindMill->maxYawError = maxYawError;
                                pWindMill->mode = "triangular";
                                break;
                            case armor::RM_WINDMILL_LARGE_ANTIC:
                                isClient.addText("RM_WINDMILL_ANTIC");
                                pWindMill->maxPitchError = -maxPitchError;
                                pWindMill->maxYawError = -maxYawError;
                                pWindMill->mode = "triangular";
                                break;
                            }



                            if (pWindMill->run(frame, pitch, yaw, (double)cv::getTickCount())) {//有目标，运行风车击打
                                communicator.send(0.0, 0.0,
                                    armor::SEND_STATUS_WM_AIM, armor::SEND_STATUS_WM_FIND);
                                isClient.addText(cv::format("send pitch:%0.2f", pitch));
                                isClient.addText(cv::format("send yaw:%0.2f", yaw));


                            }
                            else {//无目标
                                communicator.send(0.0, 0.0,
                                    armor::SEND_STATUS_WM_AIM, armor::SEND_STATUS_WM_NO);
                                PRINT_WARN("[windmill] no target find\n");

                            }
                            isClient.addText(cv::format("delay: %0.2f", pWindMill->delay));
                            isClient.addText(cv::format("maxPitchError: %0.2f", pWindMill->maxPitchError));
                            isClient.addText(cv::format("maxYawError: %0.2f", pWindMill->maxYawError));
                            isClient.show();
                        }
                        break;
                    case armor::RM_AUTO_ATTACK://跑自瞄
                        if (attack.run(frame, timeStamp, gYaw, gPitch))
                            /* 通知主线程显示图像, 有时候这一帧放弃的话就不显示了 */
                            isClient.show();
                        break;
                    default:
                        break;
                    }

                    static std::chrono::high_resolution_clock::time_point beg, end;
                    static double now_max = 0;
                    end = std::chrono::high_resolution_clock::now();
                    auto cost = std::chrono::duration<double, std::milli>((end - beg)).count();
                    beg = std::chrono::high_resolution_clock::now();
                    if (cost < 10000)
                        now_max = std::max(now_max, cost);
                    std::cout << "@@@@@@@@@@@@@@@ " << cost << " ms\n";
                    std::cout << "@@@@@@@@@@@@@@@ NOW MAX -> " << now_max << " ms\n";
                    isClient.clock("run");

                }
                else {
                    PRINT_ERROR("capture wait_and_get() failed once\n");
                }

                while (isServer.isPause()) { armor::thread_sleep_us(5); }
            }
            PRINT_INFO("attackThreads %d quit \n", i);

            });
    }

    // 图像显示主循环
    /*
    include imageshow.hpp
    ImageShowServer
    m_keyEvent捕获用户按键输入
    m_createModifiedWindow()生成调整过位置和大小的窗口
    mainloop（）在图像上执行绘图以及交互等操作，进行图像显示
    */
    isServer.mainloop();
    for (auto& _t : attackThreads) _t.join();

#ifndef USE_USB
    communicator.letStop();
    communicator.join();
#endif

    std::cout << "Attack End" << std::endl;
    exit(0);
    return 0;
}