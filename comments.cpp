 #include "attack.hpp"
#include <windmill/Windmill.hpp>
#include "capture.hpp"
#include "communicator.hpp"
#include "imageshow.hpp"

int main()
{
    std::cout << "Using OpenCV " << CV_VERSION << std::endl;
    int threadNum = armor::stConfig.get<int>("auto.thread-num");//�߳�����ʼ��

    /* ͨ�� */
#ifdef USE_USB
    armor::CommunicatorUSB communicator;
    communicator.open(0x0477, 0x5620);
#else
    /*
    ���ô���ͨ��:
    ���豸 open()(���ػ��̺߳�ת��ʱ��Ƭ��
    ��Ϣ���� send()   
    ��ֵУ����� startReceiveService()����ʼ�����̣߳�
    */
    armor::CommunicatorSerial communicator;
    communicator.disable(!armor::stConfig.get<bool>("communicator.enable"));
    communicator.open(armor::stConfig.get<std::string>("communicator.serial-port"));
#endif
    communicator.startReceiveService(); //��ʼ�����߳�

    /*����ͷ����
      init()��ʼ��
      play()��ʼ����ͼ��
      setCaotureROI()����ROI
      wait_and_get()�����Ի�ȡ����ͷ
      getCurrentInterval()���زɼ�ʱ����
    */
    
    armor::Capture* cap = nullptr;
    armor::LajiVision lajiVision;
    armor::DaHuaVision dahuaVision;

    if (armor::stConfig.get<bool>("cap.is-video")) {//����Ƶ������������ͷ
        cap = &lajiVision;
    }
    else {
        cap = &dahuaVision;//�򿪴�����ͷ
    }

    cap->init(); // ��ʼ��

    if (armor::stConfig.get<bool>("cap.is-video")) {
        cap->setCaptureROI(cv::Size2i(1280, 900), armor::CAP_ROI_CENTER_CENTER); // �ɼ�ROI
    }
    else {
        cap->setCaptureROI(cv::Size2i(1280, 900), armor::CAP_ROI_CENTER_CENTER); // �ɼ�ROI
    }
    cap->play(); //��ʼ�ɼ�ͼ��

    //include imageshow.hpp  
    /* ��ͼ����ʾ��������
       setMode()����ģʽ
       setFontSize()��������
       enableClockPrint()���clock��ʱ
       enableAverageCostPrint()���CPUƽ����ʱ
     
    */
    armor::ImageShowServer isServer(threadNum, 0.5);//����
   //����ģʽ�����֣����忴�����ļ�
    isServer.setMode(armor::stConfig.get<int>("isServer.mode"));//����ģʽ�����֣����忴�����ļ�
    isServer.setFontSize(1.25);
    isServer.enableClockPrint(true);
    isServer.enableAverageCostPrint(true);

    /* attack�߳��� */
    std::vector <std::thread> attackThreads;
    attackThreads.resize(threadNum);
    armor::PID pid;//��صģ������ù�
    pid.init(armor::stConfig.get<double>("auto.kp"),
        armor::stConfig.get<double>("auto.ki"),
        armor::stConfig.get<double>("auto.kd"));

    for (int i = 0; i < threadNum; ++i) {//����ÿ���̶߳����attack��windmill���г�ʼ����֮������ѭ����i��ֵ��Ϊ��ͬ���̷߳��䲻ͬ����������������߷糵����
        attackThreads[i] = std::thread([cap, &isServer, i, &communicator, &pid]() {
            armor::ImageShowClient isClient = isServer.getClient(i);//include imageshow.hpp

        /* ��ʼ�� attack 
               modeΪRM_AUTO_ATTACKʱִ������
               ��ʼ����..��������Ĵ�����Щû����
               �������������
               m_preDetect--Ԥ������װ�װ壨hsv�����ͣ�����ȣ�����ȣ�
               mat2tensor--Ԥ����õ�Mat ת��Ϊtensor��
               getThreshold--�õ���ֵ����ֵ
               init_my_tf������
               m_classify_single_tensor--������
               m_match--����--ѡ�õ�װ�װ���λ�
               getBoundingRect--ͼ����չROI���ı仭���С
               run��
                   __������߳��¾�������Ҫ���� 
                   __m_preDetect
                   __m_classsify_single_tensor
                   __m_preTargets���Ԥ������
                   __Ŀ��ƥ��+Ԥ��+��������+����ŷ����+�������
         */

            armor::Attack attack(communicator, pid, isClient);
            attack.enablePredict(armor::stConfig.get<bool>("auto.enable-predict"));//�Ƿ����Ԥ��

            attack.setMode(armor::stConfig.get<std::string>("attack.attack-color") == "red");//������ɫΪ��ɫ������������ļ�

            /* ��ʼ�� windmill 
               modeΪRM_WINDMILLʱִ�з糵
               ���������
               Ԥ����
               ɸѡ
               pnp��Ƕ��Լ�����
               ʵ������0�߳��ܷ糵
            */
            cv::Mat TvCtoL = (cv::Mat_<double>(3, 1) << armor::stConfig.get<double>("power.x"), armor::stConfig.get<double>("power.y"), armor::stConfig.get<double>("power.z"));  //����ͷ����̨ת������


            double delay = armor::stConfig.get<double>("power.delay");
            double maxPitchError = armor::stConfig.get<double>("power.maxPitchError");
            double maxYawError = armor::stConfig.get<double>("power.maxYawError");


            /* �糵ʵ���� 
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

                    /* ˢ�����̴߳���ͼ�� */
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
                        // ָ�������߳�
                        if (i == 0) {//i=0���߳��ܷ糵
                            /* ��糵 */
                            float pitch = 0.0;
                            float yaw = 0.0;
                            switch (mode) {//ѡ��ģʽ��ʼ�����������Ǿɰ棬���°�δ���£�
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



                            if (pWindMill->run(frame, pitch, yaw, (double)cv::getTickCount())) {//��Ŀ�꣬���з糵����
                                communicator.send(0.0, 0.0,
                                    armor::SEND_STATUS_WM_AIM, armor::SEND_STATUS_WM_FIND);
                                isClient.addText(cv::format("send pitch:%0.2f", pitch));
                                isClient.addText(cv::format("send yaw:%0.2f", yaw));


                            }
                            else {//��Ŀ��
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
                    case armor::RM_AUTO_ATTACK://������
                        if (attack.run(frame, timeStamp, gYaw, gPitch))
                            /* ֪ͨ���߳���ʾͼ��, ��ʱ����һ֡�����Ļ��Ͳ���ʾ�� */
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

    // ͼ����ʾ��ѭ��
    /*
    include imageshow.hpp
    ImageShowServer
    m_keyEvent�����û���������
    m_createModifiedWindow()���ɵ�����λ�úʹ�С�Ĵ���
    mainloop������ͼ����ִ�л�ͼ�Լ������Ȳ���������ͼ����ʾ
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