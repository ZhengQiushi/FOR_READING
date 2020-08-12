//
// Created by truth on 2020/2/15.
//

#ifndef TJSP_ATTACK_2020_ATTACK_HPP
#define TJSP_ATTACK_2020_ATTACK_HPP

#include <numeric>

#include "base.hpp"
#include "capture.hpp"
#include "imageshow.hpp"
#include "communicator.hpp"
#include "ThreadPool.h"
#include <thread>
#include <future>
#include "layers.hpp"
#include <utility>
#include <dirent.h>

//tf
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::string;
using tensorflow::Tensor;

using namespace tensorflow;

/*模型路径*/
const string model_path = "../Model/happyModel.pb";
/*输入输出节点详见ipynb的summary*/
const string input_name = "input_1:0";
const string output_name = "y/Sigmoid:0";
//const int fixedSize=32;
//#define BLUE

namespace armor
{

class Hog
{
private:
    cv::Size m_winSize;
    cv::HOGDescriptor m_hog;

    void m_preProcess(cv::Mat &_crop)
    {
        CV_Assert(_crop.cols >= m_winSize.width && _crop.rows >= m_winSize.height);
        if (_crop.rows > 64)
            cv::resize(_crop, _crop, cv::Size2i(64, 64), cv::INTER_AREA);
        else
            cv::resize(_crop, _crop, cv::Size2i(64, 64), cv::INTER_CUBIC);
        cv::GaussianBlur(_crop, _crop, cv::Size(3, 3), -1);
    }

public:
    explicit Hog() : m_winSize(cv::Size(64, 32)), m_hog(
                                                      cv::HOGDescriptor(m_winSize, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9))
    {
    }

    /**
         * 计算单张图的hog
         * @param img 图
         * @param gradient 计算结果
         */
    void computeHOG_Single(cv::Mat &img, cv::Mat &gradient)
    {
        /* 预处理 */
        m_preProcess(img);
        /* 计算hog */
        std::vector<float> descriptors;
        m_hog.compute(img, descriptors, cv::Size(8, 8), cv::Size(0, 0));
        gradient = cv::Mat(descriptors).clone();
    }

    /**
         * 计算多张图的hog
         * @param img_lst 图集合
         * @param gradient_lst 计算结果集合
         */
    void computeHOGs(std::vector<cv::Mat> &img_lst, std::vector<cv::Mat> &gradient_lst)
    {
        std::vector<float> descriptors;
        for (auto &img : img_lst)
        {
            /* 预处理 */
            m_preProcess(img);
            /* 计算hog */
            m_hog.compute(img, descriptors, cv::Size(8, 8), cv::Size(0, 0));
            gradient_lst.emplace_back(cv::Mat(descriptors).clone());
        }
    }

    /**
         * std::vector<cv::Mat>转cv::Mat
         * @param train_samples
         * @param trainData
         */
    void convert_to_ml(const std::vector<cv::Mat> &train_samples, cv::Mat &trainData)
    {
        const int rows = (int)train_samples.size();
        const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
        cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
        trainData = cv::Mat(rows, cols, CV_32FC1);
        for (size_t i = 0; i < train_samples.size(); ++i)
        {
            CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);
            if (train_samples[i].cols == 1)
            {
                transpose(train_samples[i], tmp);
                tmp.copyTo(trainData.row((int)i));
            }
            else if (train_samples[i].rows == 1)
            {
                train_samples[i].copyTo(trainData.row((int)i));
            }
        }
    }
};

/**
     * 自瞄基类, 多线程共享变量用
     */
class AttackBase
{
protected:
    static std::mutex s_mutex;
    static std::atomic<int64_t> s_latestTimeStamp; // 已经发送的帧编号
    static std::deque<Target> s_historyTargets;  // 打击历史, 最新的在头部, [0, 1, 2, 3, ....]
    static Kalman kalman;
};

std::mutex AttackBase::s_mutex;
std::atomic<int64_t> AttackBase::s_latestTimeStamp(0);
std::deque<Target> AttackBase::s_historyTargets;
Kalman AttackBase::kalman;

/**
     * 自瞄主类
     */
class Attack : AttackBase
{
private:
    cv::Mat m_bgr;
    cv::Mat m_bgr_raw;

    Communicator &m_communicator;

    ImageShowClient &m_is;
    cv::Ptr<cv::MSER> m_mser;
    Hog m_hog;

    cv::Ptr<cv::ml::RTrees> m_forest;
    // 目标
    std::vector<Target> m_preTargets; // 预检测目标
    std::vector<Target> m_targets;    // 本次有效目标集合
    // 开小图
    cv::Point2i m_startPt;
    bool m_isEnablePredict; // 是否开预测

    int64_t m_currentTimeStamp;
    PID &m_pid;
    bool m_isUseMSER;
    bool m_isUseDialte;

    bool mode = 0; // RED:0  BLUE:1

public:
    explicit Attack(Communicator &communicator, PID &pid, ImageShowClient &isClient) : m_communicator(communicator),
                                                                                       m_is(isClient),
                                                                                       m_mser(cv::MSER::create(5, 5, 800, 0.7)),
                                                                                       //  m_forest(cv::ml::RTrees::load("../data/rf.xml")),
                                                                                       m_isEnablePredict(true), m_currentTimeStamp(0), m_pid(pid), m_isUseMSER(false), m_isUseDialte(false)
    {
        mycnn::loadWeights("../info/dumpe2.nnet");
        m_isUseMSER = stConfig.get<bool>("auto.is-mser");
        m_isUseDialte = stConfig.get<bool>("auto.is-dilate");
    }

    void setMode(bool colorMode) { mode = colorMode; }

private:
    // mser重叠区域剔除
    float m_overlap(const cv::Rect &box1, const cv::Rect &box2) noexcept
    {
        if (box1.x > box2.x + box2.width || box1.y > box2.y + box2.height || box1.x + box1.width < box2.x ||
            box1.y + box1.height < box2.y)
        {
            return 0.0;
        } //此情况无重叠

        float colInt = std::min(box1.x + box1.width, box2.x + box2.width) - std::max(box1.x, box2.x);
        float rowInt = std::min(box1.y + box1.height, box2.y + box2.height) - std::max(box1.y, box2.y);

        float intersection = colInt * rowInt;
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float minArea = area1 < area2 ? area1 : area2;
        return intersection / minArea;
    }

    void m_preDetect()
    {
        DEBUG("m_preDetect")
        // 颜色筛选 todo:参数
        cv::Mat bgrChecked;
        if (m_isUseMSER)
        {
            cv::cvtColor(m_bgr, bgrChecked, cv::COLOR_BGR2GRAY);
            m_is.clock("mser");
            /* MSER */
            int imWidth = bgrChecked.size().width;
            int imHeight = bgrChecked.size().height;
            std::vector<std::vector<cv::Point>> regions;
            std::vector<cv::Rect> bBoxes;
            m_mser->setPass2Only(true);
            m_mser->detectRegions(bgrChecked, regions, bBoxes);
            m_is.clock("mser");
            m_is.clock("too small");
            /* 去太小 */
            std::vector<int> ids;
            for (int i = 0; i < regions.size(); i++)
            {
                cv::Rect &box = bBoxes[i];
                if (box.x < 4 || box.y < 4 || ((box.x + box.width) > (imWidth - 4)) ||
                    ((box.y + box.height) > (imHeight - 4)))
                    continue;
                if (regions[i].size() < 3)
                    continue; // 去太小,
                ids.emplace_back(i);
            }
            m_is.clock("too small");

            m_is.clock("Overlap");
            std::vector<int> idsNoRepeat;
            while (!ids.empty())
            {
                idsNoRepeat.emplace_back(ids.back());
                ids.pop_back();
                for (auto iter = ids.begin(); iter != ids.end();)
                {
                    if (m_overlap(bBoxes[*iter], bBoxes[idsNoRepeat.back()]) > 0.9)
                    {
                        iter = ids.erase(iter);
                    }
                    else
                        iter++;
                }
            }
            m_is.clock("Overlap");

            m_is.clock("Color Filter");
            cv::Mat hsv;
            cv::cvtColor(m_bgr, hsv, cv::COLOR_BGR2HSV);
            std::vector<int> idsColorFilter;
            for (const auto &_id : idsNoRepeat)
            {
                // printf("area: %d \n", regions[_id].size());
                int okPtCount = 0;
                auto okMinCount = regions[_id].size() * 0.3;
                okMinCount = okMinCount > 100 ? 100 : okMinCount;
                okMinCount = okMinCount < 1 ? 1 : okMinCount;
                bool colorFound = false;
                if (mode)
                {
                    for (const auto &_reg : regions[_id])
                    {
                        if (okPtCount >= okMinCount)
                        {
                            colorFound = true;
                            break;
                        }
                        if (hsv.at<cv::Vec3b>(_reg)[0] < 15 || hsv.at<cv::Vec3b>(_reg)[0] > 165)
                        {
                            okPtCount += 1;
                        }
                    }
                    if (!colorFound)
                        continue;
                }
                else
                {
                    for (const auto &_reg : regions[_id])
                    {
                        if (okPtCount >= okMinCount)
                        {
                            colorFound = true;
                            break;
                        }
                        if (hsv.at<cv::Vec3b>(_reg)[0] > 65 && hsv.at<cv::Vec3b>(_reg)[0] < 180 &&
                            //hsv.at<cv::Vec3b>(_reg)[1] > 6 &&
                            m_bgr.at<cv::Vec3b>(_reg)[0] > 175
                            // &&
                            // cv::abs(frame.at<cv::Vec3b>(_reg)[0] - frame.at<cv::Vec3b>(_reg)[2]) > 28
                        )
                        {
                            okPtCount += 1;
                        }
                    }
                    if (!colorFound)
                        continue;
                }
                idsColorFilter.emplace_back(_id);
            }
            bgrChecked = 0;
            for (const auto &_id : idsColorFilter)
            {
                for (const auto &point : regions[_id])
                {
                    cv::circle(bgrChecked, point, 1, (255, 255, 255), 1);
                }
            }
            m_is.clock("Color Filter");
        }
        else
        {
            m_is.clock("inRange");
            if (mode)
            {
                /* 红色 */
                cv::inRange(m_bgr, cv::Scalar(0, 0, 140), cv::Scalar(255, 255, 255), bgrChecked);
            }
            else
            {
                /* 蓝色 */
                cv::inRange(m_bgr, cv::Scalar(130, 100, 0), cv::Scalar(255, 255, 65), bgrChecked);
            }
            m_is.clock("inRange");
            DEBUG("inRange end")
        }
        m_is.addImg("bgrChecked", bgrChecked, true);
        if (m_isUseDialte)
        {
            cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            dilate(bgrChecked, bgrChecked, element);
            m_is.addImg("dilate", bgrChecked, true);
        }

        /* 寻找边缘 */
        std::vector<std::vector<cv::Point2i>> contours;
        cv::findContours(bgrChecked, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        m_is.addEvent("contours", contours);
        DEBUG("findContours end")

        std::vector<Light> lights;
        for (const auto &_pts : contours)
        {
            // 最小面积 todo:参数
            if (_pts.size() < 5)
                continue;

            cv::RotatedRect rRect = cv::minAreaRect(_pts);
            // 长宽比 todo:参数
            //            printf("area// = %.2f \n", area / rRect.size.width / rRect.size.height);
            double hw = rRect.size.height / rRect.size.width;
            if (0.6667 < hw && hw < 1.5)
                continue;

            Light _light;
            cv::Point2f topPt;
            cv::Point2f bottomPt;
            cv::Point2f pts[4];
            rRect.points(pts);
            if (rRect.size.width > rRect.size.height)
            {
                bottomPt = (pts[2] + pts[3]) / 2.0;
                topPt = (pts[0] + pts[1]) / 2.0;
                _light.angle = cv::abs(rRect.angle);
            }
            else
            {
                bottomPt = (pts[1] + pts[2]) / 2;
                topPt = (pts[0] + pts[3]) / 2;
                _light.angle = cv::abs(rRect.angle - 90);
            }
            if (topPt.y > bottomPt.y)
            {
                _light.topPt = bottomPt;
                _light.bottomPt = topPt;
            }
            else
            {
                _light.topPt = topPt;
                _light.bottomPt = bottomPt;
            }
            _light.centerPt = rRect.center;
            _light.length = cv::norm(bottomPt - topPt);

            // 筛长度和角度 todo:参数
            //            printf("length = %.2f, angle = %.2f \n", _light.length, _light.angle);
            if (_light.length < 3.0 || 800.0 < _light.length || cv::abs(_light.angle - 90) > 30.0)
                continue;
            lights.emplace_back(_light);
            //cv::line(m_bgr, _light.topPt, _light.bottomPt, cv::Scalar(255, 255, 120), 2);
        }
        DEBUG("lights end")
        m_is.addEvent("lights", lights);

        std::sort(lights.begin(), lights.end(), [](Light &a_, Light &b_) -> bool {
            return a_.centerPt.x < b_.centerPt.x;
        });

        for (size_t i = 0; i < lights.size(); ++i)
        {
            for (size_t j = i + 1; j < lights.size(); ++j)
            {
                cv::Point2f AC2BC = lights[j].centerPt - lights[i].centerPt;
                double minLength = cv::min(lights[i].length, lights[j].length);
                double deltaAngle = cv::abs(lights[i].angle - lights[j].angle);

                //                    printf("%.2f, %.2f, %.2f, %.2f, %.2f\n", deltaAngle, minLength,
                //                           cv::abs(lights[i].length - lights[j].length) / minLength,
                //                           cv::fastAtan2(cv::abs(AC2BC.y), cv::abs(AC2BC.x)), AC2BC.x / minLength);

                if ((deltaAngle > 23.0 && minLength < 20) || (deltaAngle > 11.0 && minLength >= 20) ||
                    cv::abs(lights[i].length - lights[j].length) / minLength > 0.5 ||
                    cv::fastAtan2(cv::abs(AC2BC.y), cv::abs(AC2BC.x)) > 25.0 ||
                    AC2BC.x / minLength > 7.1)
                    continue;

                //                    printf("pass \n");
                Target target;
                /* 设置像素坐标 */
                target.setPixelPts(lights[i].topPt, lights[i].bottomPt, lights[j].bottomPt, lights[j].topPt,
                                   m_startPt);
                if (cv::norm(AC2BC) / minLength > 4.9)
                    target.type = TARGET_LARGE; // 大装甲
                /* 获得扩展区域像素坐标, 若无法扩展则放弃该目标 */
                if (!target.convert2ExternalPts2f())
                    continue;
                m_preTargets.emplace_back(target);
                //target.
            }
        }
        m_is.addEvent("preTargets", m_preTargets);
        DEBUG("preTargets end")
    }

    int m_cropNameCounter = 0;

    /**
         * 分类
         * @param isSave 是否需要保存预检测样本
         */
    void m_classify(bool isSave = false)
    {

        if (m_preTargets.empty())
            return;
        ThreadPool thread_pool(10);
        static std::vector<std::future<mytype>> future_results;
        future_results.clear();

        int i = 0;
        for (auto &_tar : m_preTargets)
            future_results.emplace_back(thread_pool.enqueue([&]() {

                cv::Rect tmp = cv::boundingRect(_tar.pixelPts2f_Ex);
                cv::Mat tmp2 = m_bgr_raw(tmp).clone();

                /*变成目标大小*/
                cv::Mat transMat = cv::getPerspectiveTransform(_tar.pixelPts2f_Ex,
                                                               _tar.pixelPts2f_Ex);
                //cv::Mat transMat = cv::getPerspectiveTransform(_tar.pixelPts2f_Ex,
                //                                               armor::stArmorStdFigure.smallFig_Ex);
                cv::Mat _crop;
                /* 投影变换 */
                cv::warpPerspective(tmp2, _crop, transMat, cv::Size(tmp2.size())); //tmp2.size()
                //cv::warpPerspective(m_bgr_raw, _crop, transMat, cv::Size(armor::stArmorStdFigure.smallFig_Ex[2]));
                /* 转灰度图 */
                cv::cvtColor(_crop, _crop, cv::COLOR_BGR2GRAY);
                /* 储存图 */
                if (isSave)
                {
                    cv::imwrite(cv::format("../data/raw/%d.png", m_cropNameCounter++), _crop);
                }
                if (mycnn::loadData(_crop) != 0)
                {
                    mytype final = mycnn::run();
                    return final;
                    //if(final>0.5){
                    //_tar.calcWorldParams();
                    //m_targets.emplace_back(_tar);
                }
            }));

        assert(future_results.size() == m_preTargets.size());

        /* 分类结果处理 */
        for (size_t j = 0; j < future_results.size(); ++j)
        {
            if (future_results[j].get() <= 0.5) // 找到了!!!
            {
                //m_preTargets[j].calcWorldParams();
                m_targets.emplace_back(m_preTargets[j]);
            }
        }
        m_is.addClassifiedTargets("After Classify", m_targets);
        DEBUG("m_classify end")
    }


    void m_classify_single(bool isSave = false)
    {
        if (m_preTargets.empty())
            return;
        for (auto &_tar : m_preTargets)
        {
            cv::Rect tmp = cv::boundingRect(_tar.pixelPts2f_Ex);
            cv::Mat tmp2 = m_bgr_raw(tmp).clone();

            /*变成目标大小*/
            cv::Mat transMat = cv::getPerspectiveTransform(_tar.pixelPts2f_Ex,
                                                           _tar.pixelPts2f_Ex);
            //cv::Mat transMat = cv::getPerspectiveTransform(_tar.pixelPts2f_Ex,
            //                                               armor::stArmorStdFigure.smallFig_Ex);
            cv::Mat _crop;
            /* 投影变换 */
            cv::warpPerspective(tmp2, _crop, transMat, cv::Size(tmp2.size())); //tmp2.size()
            //cv::warpPerspective(m_bgr_raw, _crop, transMat, cv::Size(armor::stArmorStdFigure.smallFig_Ex[2]));
            /* 转灰度图 */
            cv::cvtColor(_crop, _crop, cv::COLOR_BGR2GRAY);
            /* 储存图 */
            if (isSave)
            {
                cv::imwrite(cv::format("../data/raw/%d.png", m_cropNameCounter++), _crop);
            }
            if (mycnn::loadData(_crop) != 0)
            {
                mytype final = mycnn::run();
                if (final > 0.5)
                {
                    //_tar.calcWorldParams();
                    m_targets.emplace_back(_tar);
                }
            }
        }

        m_is.addClassifiedTargets("After Classify", m_targets);
        DEBUG("m_classify end")
    }

    void mat2Tensor(cv::Mat &image, Tensor &t) {
    float *tensor_data_ptr = t.flat<float>().data();
    cv::Mat fake_mat(image.rows, image.cols, CV_32FC(image.channels()), tensor_data_ptr);
    image.convertTo(fake_mat, CV_32FC(image.channels()));
    }

    int getThreshold(const cv::Mat& mat,double thre_proportion=0.1){

    uint32_t iter_rows = mat.rows;
    uint32_t iter_cols = mat.cols;
    auto sum_pixel = iter_rows * iter_cols;
    if(mat.isContinuous()){
        iter_cols = sum_pixel;
        iter_rows = 1;
    }
    int histogram[256];
    memset(histogram, 0, sizeof(histogram));//置零
    for (uint32_t i = 0; i < iter_rows; ++i){
        const auto* lhs = mat.ptr<uchar>(i);
        for (uint32_t j = 0; j < iter_cols; ++j)
            ++histogram[*lhs++];
    }

    auto left = thre_proportion * sum_pixel;
    int i = 255;
    while((left -= histogram[i--]) > 0);
    return i>0?i:0;
    }

    bool loadAndPre(cv::Mat img,cv::Mat &result){
        //注意已经灰度化了
        //cout<<img.cols<<" "<<img.rows<<endl;
        if(img.cols==0)
            return false;
        //调整大小 同比缩放至fixedsize*fixedsize以内
        if(img.cols<img.rows)
            resize(img,img,{int(img.cols*1.0/img.rows*fixedSize),fixedSize});
        else
            resize(img,img,{fixedSize,int(img.rows*1.0/img.cols*fixedSize)});

        //剪去边上多余部分
        int cutRatio1=0.15*img.cols;
        int cutRatio2=0.05*img.rows;
        cv::Mat blank=cv::Mat(cv::Size(fixedSize,fixedSize), img.type(), cv::Scalar(0));//新建空白
        cv::Mat mask=img(cv::Rect(cutRatio1,cutRatio2,img.cols-2*cutRatio1,img.rows-2*cutRatio2));//建立腌摸
        cv::Mat imageROI=blank(cv::Rect(cutRatio1,cutRatio2,img.cols-2*cutRatio1,img.rows-2*cutRatio2));//建立需要覆盖区域的ROI
        mask.copyTo(imageROI, mask);

        //imshow("mask",mask);//小图
        //imshow("blank",blank);//大图

        int thre=getThreshold(blank);//均值获取阈值
        result=blank.clone();
        //补高光，而不直接粗暴二值化
        for (int i = 0; i<result.rows; i++){
            for (int j = 0; j<result.cols; j++){
                if((int)result.at<u_char>(i, j)>thre){
                    result.at<u_char>(i, j)=200;
                }
            }
        }
        //imshow("result",result);
        //cv::waitKey();
        return true;
    }

    inline Tensor init_my_tf(Session* session){
        /*--------------------------------从pb文件中读取模型--------------------------------*/

        GraphDef graph_def;
        //读取Graph, 如果是文本形式的pb,使用ReadTextProto
        Status status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl;
        } else {
            std::cout << "Load graph protobuf successfully" << std::endl;
        }
        /*--------------------------------将模型设置到创建的Session里--------------------------------*/
        status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl;
        } else {
            std::cout << "Add graph to session successfully" << std::endl;
        }

        Tensor input(DT_FLOAT, TensorShape({ 1, fixedSize, fixedSize, 1 }));
        return input;

    }    

    

    void m_classify_single_tensor(bool isSave = false)
    {
        if (m_preTargets.empty())
            return;
        Session* session;
        /*--------------------------------创建session------------------------------*/
        Status status = NewSession(SessionOptions(), &session);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl;
        } else {
            std::cout << "Session created successfully" << std::endl;
        }
        /*初始化*/
        Tensor input=init_my_tf(session);

        for (auto &_tar : m_preTargets)
        {
            cv::Rect tmp = cv::boundingRect(_tar.pixelPts2f_Ex);
            cv::Mat tmp2 = m_bgr_raw(tmp).clone();

            /*变成目标大小*/
            cv::Mat transMat = cv::getPerspectiveTransform(_tar.pixelPts2f_Ex,
                                                           _tar.pixelPts2f_Ex);
            //cv::Mat transMat = cv::getPerspectiveTransform(_tar.pixelPts2f_Ex,
            //                                               armor::stArmorStdFigure.smallFig_Ex);
            cv::Mat _crop;
            /* 投影变换 */
            cv::warpPerspective(tmp2, _crop, transMat, cv::Size(tmp2.size())); //tmp2.size()
            //cv::warpPerspective(m_bgr_raw, _crop, transMat, cv::Size(armor::stArmorStdFigure.smallFig_Ex[2]));
            /* 转灰度图 */
            cv::cvtColor(_crop, _crop, cv::COLOR_BGR2GRAY);
            /* 储存图 */
            if (isSave)
            {
                cv::imwrite(cv::format("../data/raw/%d.png", m_cropNameCounter++), _crop);
            }

            
            cv::Mat image;
            if(loadAndPre(_crop,image)){
                //形式的转换
                mat2Tensor(image, input);
                /*保留最终输出*/
                std::vector<tensorflow::Tensor> outputs;
                // 3、计算最后结果
                TF_CHECK_OK(session->Run({std::pair<string, Tensor>(input_name, input)}, {output_name}, {}, &outputs));
                //获取输出
                auto output_c = outputs[0].scalar<float>();
                float result = output_c();
                /*判断正负样本*/
                if(0.5<result){
                    m_targets.emplace_back(_tar);
                }
            }
            else
                continue;

            
        }
        session->Close();
        m_is.addClassifiedTargets("After Classify", m_targets);
        DEBUG("m_classify end")
    }
    emSendStatusA m_match()
    {
        /* 更新下相对帧编号 */
        for (auto iter = s_historyTargets.begin(); iter != s_historyTargets.end(); iter++)
        {
            iter->rTick++;
            // TODO: 历史值数量参数
            if (iter->rTick > 30)
            {
                s_historyTargets.erase(iter, s_historyTargets.end());
                break;
            }
        }
        /* 选择本次打击目标 */
        if (s_historyTargets.empty())
        {
            /* case A: 之前没选择过打击目标 */
            auto minTarElement = std::min_element(
                m_targets.begin(), m_targets.end(), [](Target &a_, Target &b_) -> bool {
                    return cv::norm(a_.ptsInGimbal) < cv::norm(b_.ptsInGimbal);
                });
            if (minTarElement != m_targets.end())
            {
                s_historyTargets.emplace_front(*minTarElement);
                PRINT_INFO("++++++++++++++++ 发现目标: 选择最近的 ++++++++++++++++++++\n");
                return SEND_STATUS_AUTO_AIM;
            }
            else
            {
                return SEND_STATUS_AUTO_NOT_FOUND;
            }
        }
        else
        {
            /* case B: 之前选过打击目标了, 得找到一样的目标 */
            PRINT_INFO("++++++++++++++++ 开始寻找上一次目标 ++++++++++++++++++++\n");
            double distance = 0xffffffff;
            int closestElementIndex = -1;
            for (size_t i = 0; i < m_targets.size(); ++i)
            {
                // 轮廓匹配, 0~1, 0 = 一模一样
                double distanceA = cv::matchShapes(m_targets[i].pixelPts2f, s_historyTargets[0].pixelPts2f,
                                                   cv::CONTOURS_MATCH_I3, 0.0);
                cv::Moments m_1 = cv::moments(m_targets[i].pixelPts2f);
                cv::Moments m_2 = cv::moments(s_historyTargets[0].pixelPts2f);
                PRINT_WARN("distanceA = %f\n", distanceA);
                // TODO: 轮廓匹配阈值, 参数修改
                if (distanceA > 0.5 ||
                    (m_1.nu11 + m_1.nu30 + m_1.nu12) * (m_2.nu11 + m_2.nu30 + m_2.nu12) < 0)
                    continue;

                double distanceB;
                if (m_isEnablePredict)
                {
                    /* 用绝对坐标距离计算 TODO: 绝对坐标距离阈值, 参数修改 */
                    distanceB = cv::norm(m_targets[i].ptsInWorld - s_historyTargets[0].ptsInWorld) / 2000.0;
                    PRINT_WARN("distanceB = %f\n", distanceB);
                    if (distanceB > 0.5)
                        continue;
                }
                else
                {
                    /* 用云台坐标系距离计算 TODO: 像素距离阈值, 参数修改 */
                    distanceB = cv::norm(m_targets[i].ptsInGimbal - s_historyTargets[0].ptsInGimbal) / 3400.0;
                    PRINT_WARN("distanceB = %f\n", distanceB);
                    if (distanceB > 0.8)
                        continue;
                }
                double _distanceTemp = distanceA + distanceB / 2; // TODO: 改参数
                if (distance > _distanceTemp)
                {
                    distance = _distanceTemp;
                    closestElementIndex = i;
                }
            }
            if (closestElementIndex != -1)
            {
                /* 找到了 */
                s_historyTargets.emplace_front(m_targets[closestElementIndex]);
                PRINT_INFO("++++++++++++++++ 找到上一次目标 ++++++++++++++++++++\n");
                return SEND_STATUS_AUTO_AIM;
            }
            else
            {
                PRINT_INFO("++++++++++++++++ 没找到上一次目标, 按上一次的来 ++++++++++++++++++++\n");
                return SEND_STATUS_AUTO_AIM_FORMER; // 没找到按上一次的来
            }
        } // end case B
        PRINT_ERROR("Something is NOT Handled in function m_match \n");
    }

public:
    /**
         * 设置是否开启预测
         * @param enable = true: 开启
         */
    void enablePredict(bool enable = true)
    {
        m_communicator.enableReceiveGlobalAngle(enable);
        m_isEnablePredict = enable;
    }

    /**
         * 开小图
         * @param tar 上一个检测到的装甲
         * @param rect 截的图
         * @param extendFlag 是否扩展
         */
    void getBoundingRect(Target &tar, cv::Rect &rect, cv::Size &size, bool extendFlag = false)
    {
        rect = cv::boundingRect(s_historyTargets[0].pixelPts2f_Ex);

        if (extendFlag)
        {
            rect.x -= int(rect.width * 4);
            rect.y -= rect.height * 3;
            rect.width *= 9;
            rect.height *= 7;

            rect.width = rect.width >= size.width ? size.width - 1 : rect.width;
            rect.height = rect.height >= size.height ? size.height - 1 : rect.height;

            rect.width = rect.width < 80 ? 80 : rect.width;
            rect.height = rect.height < 50 ? 50 : rect.height;

            rect.x = rect.x < 1 ? 1 : rect.x;
            rect.y = rect.y < 1 ? 1 : rect.y;

            rect.width = rect.x + rect.width >= size.width ? size.width - 1 - rect.x : rect.width;
            rect.height = rect.y + rect.height >= size.height ? size.height - 1 - rect.y : rect.height;
        }
    }

    /**
         * 主运行函数
         * @param src 彩图
         * @param timeStamp 时间戳
         */
    bool run(cv::Mat &src, int64_t timeStamp, float gYaw, float gPitch)
    {

        m_bgr_raw = src;
        m_bgr = src;
        m_currentTimeStamp = timeStamp;
        m_targets.clear();
        m_preTargets.clear();
        m_startPt = cv::Point(0, 0);
        if (s_historyTargets.size() >= 2 && s_historyTargets[0].rTick <= 10)
        {
            cv::Rect latestShootRect;
            getBoundingRect(s_historyTargets[0], latestShootRect, stFrameInfo.size, true);
            m_is.addEvent("Bounding Rect", latestShootRect);
            m_bgr = m_bgr(latestShootRect);
            m_startPt = latestShootRect.tl();
        }

        /* step 1: 预检测 */
        m_preDetect();

        /* step 2: 分类 */
        m_is.clock("m_classify");
        m_classify_single_tensor(0); //m_is.get_and_clearCurrentKey() == 's'
        m_is.clock("m_classify");

        /* 已经有更新的一帧发出去了 */
        if (timeStamp < s_latestTimeStamp.load())
            return false;

        /* 处理多线程新旧数据处理的问题 */
        std::unique_lock<std::mutex> preLock(s_mutex, std::try_to_lock);
        while (!preLock.owns_lock() && timeStamp > s_latestTimeStamp.load())
        {
            armor::thread_sleep_us(5);
            preLock.try_lock();
        }

        /* 目标匹配 + 预测 + 修正弹道 + 计算欧拉角 + 射击策略 */
        if (preLock.owns_lock() && timeStamp > s_latestTimeStamp.load())
        {
            s_latestTimeStamp.exchange(timeStamp);

            float rYaw = 0.0;
            float rPitch = 0.0;
            m_communicator.getGlobalAngle(&gYaw, &gPitch);
            for (auto &tar : m_targets)
            {
                tar.calcWorldParams();
                tar.convert2WorldPts(-gYaw, gPitch);
            }
            /* step 3: 目标匹配 */
            emSendStatusA statusA = m_match();
            DEBUG("m_match end")

            /* step 4: 修正弹道 + 计算欧拉角 + 射击策略 */
            if (!s_historyTargets.empty())
            {
                m_is.addFinalTargets("selected", s_historyTargets[0]);
                if (m_isEnablePredict)
                {
                    cout<<"m_isEnablePredict start !"<<endl;

                    if (statusA == SEND_STATUS_AUTO_AIM)
                    {
                        m_communicator.getGlobalAngle(&gYaw, &gPitch);
                        s_historyTargets[0].convert2WorldPts(-gYaw, gPitch);

                        cout<<"s_historyTargets[0].ptsInGimbal : " << s_historyTargets[0].ptsInGimbal<<endl;

                        if (s_historyTargets.size() == 1)
                            kalman.clear_and_init(s_historyTargets[0].ptsInWorld, timeStamp);
                        else{
                            kalman.correct(s_historyTargets[0].ptsInWorld, timeStamp);
                        }
                    }
                    m_is.addText(cv::format("inWorld.x %.0f", s_historyTargets[0].ptsInWorld.x));
                    m_is.addText(cv::format("inWorld.y %.0f", s_historyTargets[0].ptsInWorld.y));
                    m_is.addText(cv::format("inWorld.z %.0f", s_historyTargets[0].ptsInWorld.z));

                    if (s_historyTargets.size() > 1)
                    {
                        kalman.predict(0.1, s_historyTargets[0].ptsInWorld_Predict);

//                        std::cout<<"ptsInWorld : \n"<<s_historyTargets[0].ptsInWorld<<endl;
//                        std::cout<<"ptsInWorld_Predict : \n"<<s_historyTargets[0].ptsInWorld_Predict<<endl;
//
//
//                        /*世界坐标系转换为像素坐标系
//                         * s = M1(内参 stCamera.camMat )*M2(外参)*Xw(世界坐标)
//                         */
//                        //
//                        cv::Point3d old_pst = s_historyTargets[0].ptsInWorld;
//                        cv::Point3d new_pst = s_historyTargets[0].ptsInWorld_Predict;
//                        auto old_pstInPixel = s_historyTargets[0].pixelCenterPt2f;
//
//                        cv::Mat pstInworld = (cv::Mat_<double>(3, 1) <<old_pst.x,old_pst.y,old_pst.z);
//
//                        cv::Mat Xw;
//                        cv::Mat ptsInWorld_Predict = (cv::Mat_<double>(3, 1) <<new_pst.x,new_pst.y,new_pst.z);
//                        cv::Mat pad = (cv::Mat_<double>(1, 1) << 0);
//                        cv::vconcat(ptsInWorld_Predict,pad,Xw);
//                        std::cout<<"Xw : \n"<<Xw<<endl;
//
//                        cv::Mat M1 ;
//                        cv::Mat pad1 = (cv::Mat_<double>(3, 1) << 0,0,0);
//                        std::cout<<"camMat : \n"<<stCamera.camMat<<endl;
//                        std::cout<<"pad1 : \n"<<pad1<<endl;
//
//                        cv::hconcat(stCamera.camMat,pad1,M1);
//                        std::cout<<"M1 : \n"<<M1<<endl;
//
//                        cv::Mat M2;
//                        std::cout<<"rvMat : \n"<<s_historyTargets[0].rvMat<<endl;
//                        std::cout<<"tv : \n"<<s_historyTargets[0].tv<<endl;
//                        cv::hconcat(s_historyTargets[0].rvMat,s_historyTargets[0].tv,M2);
//                        cv::Mat pad2 = (cv::Mat_<double>(1, 4) << 0,0,0,0);
//                        cv::vconcat(M2,pad2,M2);
//                        std::cout<<"M2 : \n"<<M2<<endl;
//
//
//
//                        cv::Mat tmp = (cv::Mat_<double>(3, 1) <<old_pstInPixel.x,old_pstInPixel.y,1);
//                        std::cout<<"tmp : \n"<<tmp<<endl;
//
//
//                        cv:: Mat tempMat = s_historyTargets[0].rvMat.inv() *  stCamera.camMat.inv() * tmp;
//                        cv:: Mat tempMat2 = s_historyTargets[0].rvMat.inv() * s_historyTargets[0].tv;
//                        double s = 0 + tempMat2.at<double>(2, 0);
//                        s /= tempMat.at<double>(2, 0);
//                        std::cout<<"s : "<<s<<endl;
//                        //s = stCamera.camMat*(s_historyTargets[0].rvMat*pstInworld+s_historyTargets[0].tv)*tmp.inv();
//
//
//
//
//
//                        cv::Mat pstInPixel = M1*M2*Xw/s;
//                        std::cout<<"pstInPixel : \n "<<pstInPixel<<endl;


                        s_historyTargets[0].convert2GimbalPts(kalman.velocity);
                        m_is.addText(cv::format("vx %4.0f", s_historyTargets[0].vInGimbal3d.x));
                        m_is.addText(cv::format("vy %4.0f", cv::abs(s_historyTargets[0].vInGimbal3d.y)));
                        m_is.addText(cv::format("vz %4.0f", cv::abs(s_historyTargets[0].vInGimbal3d.z)));
                        if (cv::abs(s_historyTargets[0].vInGimbal3d.x) > 1.6)
                        {
                            double deltaX = cv::abs(13 * cv::abs(s_historyTargets[0].vInGimbal3d.x) *
                                                    s_historyTargets[0].ptsInGimbal.z / 3000);
                            deltaX = deltaX > 300 ? 300 : deltaX;
                            s_historyTargets[0].ptsInGimbal.x +=
                                deltaX * cv::abs(s_historyTargets[0].vInGimbal3d.x) /
                                s_historyTargets[0].vInGimbal3d.x;
                        }
                    }
                }

                DEBUG("correctTrajectory_and_calcEuler start")

                /* 修正弹道 + 计算欧拉角 */
                s_historyTargets[0].correctTrajectory_and_calcEuler();
                DEBUG("correctTrajectory_and_calcEuler end")

                rYaw = s_historyTargets[0].rYaw;
                rPitch = s_historyTargets[0].rPitch;

                /* 射击策略 TODO: 改参数 */
                if (s_historyTargets.size() >= 3 &&
                    cv::abs(s_historyTargets[0].ptsInShoot.x) < 70.0 &&
                    cv::abs(s_historyTargets[0].ptsInShoot.y) < 60.0 &&
                    cv::abs(s_historyTargets[1].ptsInShoot.x) < 120.0 && cv::abs(s_historyTargets[1].ptsInShoot.y) < 90.0)
                {
                    statusA = SEND_STATUS_AUTO_SHOOT;
                }

                m_is.addText(cv::format("ptsInGimbal: %2.3f %2.3f %2.3f",
                                        s_historyTargets[0].ptsInGimbal.x / 1000.0,
                                        s_historyTargets[0].ptsInGimbal.y / 1000.0,
                                        s_historyTargets[0].ptsInGimbal.z / 1000.0));

                m_is.addText(cv::format("rPitch %.3f", rPitch));
                m_is.addText(cv::format("rYaw   %.3f", rYaw));
                m_is.addText(cv::format("gYaw   %.3f", gYaw));
                m_is.addText(cv::format("rYaw + gYaw   %.3f", rYaw - gYaw));
            }

            /* PID */
            float newYaw = rYaw;
            if (cv::abs(rYaw) < 5)
            {
                newYaw = m_pid.calc(rYaw, timeStamp);
            }
            else
            {
                m_pid.clear();
            }
            m_is.addText(cv::format("newYaw %3.3f", newYaw));
            m_is.addText(cv::format("delta yaw %3.3f", newYaw - rYaw));

            newYaw = cv::abs(newYaw) < 0.3 ? rYaw : newYaw;

            /* step 6: 发给电控 */
            m_communicator.send(newYaw, rPitch, statusA, SEND_STATUS_WM_PLACEHOLDER);
            PRINT_INFO("[attack] send = %ld", timeStamp);
        }
        if (preLock.owns_lock())
            preLock.unlock();
        return true;
    }
};
} // namespace armor

#endif //TJSP_ATTACK_2020_ATTACK_HPP
