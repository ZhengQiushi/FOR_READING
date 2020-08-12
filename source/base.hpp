//
// Created by sp on 19-6-19.
//

#ifndef ATTACK_BASE_HPP
#define ATTACK_BASE_HPP

#include <opencv2/opencv.hpp>
#include <cmath>
#include <toml.h>

#include "semaphore.hpp"
//#include "capture.hpp"

#define DEBUG_MODE  // 使能 DEBUG 宏

#ifdef DEBUG_MODE

/* 文件名获取 */
#define __FILENAME__ (strrchr(__FILE__, '/') + 1)

/** __PRETTY_FUNCTION__ is only provide by gcc
 * for other compiler, use following code to capture the name of the enclosing function.
 *   auto& _func_ = __func__;
 *  [&]{ std::cout << _func_; };
 */
#define DEBUG(X) {} // cout nothing
//#define DEBUG(content)                 \
//    std::cout << "\033[35m[" << __FILENAME__ << "]" \
//         << "[" << __PRETTY_FUNCTION__ << "]" \
//         << "[" << __LINE__ << "]:" << content << " \033[0m" << std::endl;
#else
#define DEBUG(content)
#endif  

#define STATE(level, name, content) cout << "--- [" << level << "][" << name << "] " << content << endl;
#define ERROR "error"
#define INFOO "infoo"
#define WARN "warn"

/* 绿色 */
#define PRINT_INFO(content, ...) printf("\033[32m" content "\033[0m", ##__VA_ARGS__)
/* 黄色 */
#define PRINT_WARN(content, ...) printf("\033[33m" content "\033[0m", ##__VA_ARGS__)
/* 红色 */
#define PRINT_ERROR(content, ...) printf("\033[31m" content "\033[0m", ##__VA_ARGS__)

namespace armor {

    /**
     * 配置文件结构体
     */
    struct Config {
        toml::Value config;
        bool attackColor;

        explicit Config() {
            /* parse config.toml */
            std::ifstream ifs("../config.toml");
            toml::ParseResult pr = toml::parse(ifs);
            ifs.close();
            if (!pr.valid()) {
                PRINT_ERROR("[config] config toml error: %s\n", pr.errorReason.c_str());
                PRINT_ERROR("[config] abort!\n");
                exit(0);
            }
            config = pr.value;
        }

        template<typename T>
        inline typename toml::call_traits<T>::return_type get(const std::string &key) const {
            return config.get<T>(key);
        }

    } stConfig;


    /**
     * 采集的图像的参数
     */
    struct FrameInfo {
        cv::Size2i size = cv::Size2i(0, 0);
        cv::Point2i offset = cv::Point2i(0, 0);
    } stFrameInfo;

    /**
     * 相机结构体
     */
    struct Camera {
        cv::Mat camMat;
        cv::Mat distCoeffs;


        cv::Mat m_ABC_x, m_ABC_y;

        bool isTest = false;
        bool isLinear = false;

//        std::vector<double *> m_kb_x;  // x = k * z + b
//        std::vector<double *> m_kb_y;  // y = k * z + b

        /**
         * 折线弹道
         */
        struct ShootLines {

            std::vector<std::array<double, 4> > m_lhkb;

            explicit ShootLines() {}

            /**
             * y = k * x + b
             * @param y2x
             * [2000, -100]  [3000, -150], [2200, ]
             */
            void fit(std::deque<cv::Point2f> &y2x) {
                std::sort(y2x.begin(), y2x.end(), [](cv::Point2f &a, cv::Point2f &b) -> bool {
                    return a.x < b.x;
                });
                for (size_t i = 0; i < y2x.size() - 1; ++i) {
                    std::array<double, 4> _lhkb = {0, 0, 0, 0};
                    _lhkb[0] = y2x[i].x;
                    _lhkb[1] = y2x[i + 1].x;
                    _lhkb[2] = (y2x[i].y - y2x[i + 1].y) / (y2x[i].x - y2x[i + 1].x);
                    _lhkb[3] = y2x[i].y - _lhkb[2] * y2x[i].x;
                    m_lhkb.emplace_back(_lhkb);
                }
            }

            /**
             * 返回偏差值
             * @param z 弹道距离
             * @return
             */
            double getDelta(double z) {
                for (const auto &_lhkb:m_lhkb) {
                    if (z < _lhkb[1])
                        return z * _lhkb[2] + _lhkb[3];
                }
                return z * m_lhkb.back()[2] + m_lhkb.back()[3];
            }
        } m_stShootLines_x, m_stShootLines_y;

        /**
         * 结构体构造函数
         * @param path 相机参数文件路径
         */
        explicit Camera(const cv::String &path) {
            cv::FileStorage fs(path, cv::FileStorage::READ);
            cv::read(fs["camera_matrix"], camMat);
            cv::read(fs["distortion_coefficients"], distCoeffs);
            fs.release();

            /* TODO: 初始化弹道参数 */

            std::deque<cv::Point2f> disToY;
            for (auto &_pts : armor::stConfig.get<toml::Array>("curve.dy")) {
                std::vector<toml::Value> a = _pts.as<std::vector<toml::Value>>();
                cv::Point2f pt = cv::Point2f(a[0].asNumber(), a[1].asNumber());
                disToY.emplace_back(pt);
            }
            curveFit(disToY, 2, m_ABC_y);

            std::deque<cv::Point2f> disToX;
            for (auto &_pts : armor::stConfig.get<toml::Array>("curve.dx")) {
                std::vector<toml::Value> a = _pts.as<std::vector<toml::Value>>();
                cv::Point2f pt = cv::Point2f(a[0].asNumber(), a[1].asNumber());
                disToX.emplace_back(pt);
            }
            curveFit(disToX, 2, m_ABC_x);
            isTest = stConfig.get<bool>("curve.test-shoot");
            isLinear = stConfig.get<bool>("curve.use-linear");
            if (isLinear) {
                /* 使用折线拟合 */
                m_stShootLines_x.fit(disToX);
                m_stShootLines_y.fit(disToY);
            }
        }

        /**
         * Left * ABC = Right
         * z ^ 2 * A + z * B + C = dy
         * @param disToY [[x_0, y_0], [x_1, y_1], ...]
         * @param degree 多项式次数
         * @param ABC 拟合结果
         */
        static void curveFit(std::deque<cv::Point2f> &disToY, int degree, cv::Mat &ABC) {
            size_t funNum = disToY.size();
            cv::Mat Left = cv::Mat::zeros(funNum, degree + 1, CV_32F);
            cv::Mat Right = cv::Mat::ones(funNum, 1, CV_32F);
            for (size_t i = 0; i < funNum; ++i) {
                for (size_t j = 0; j < degree + 1; ++j) {
                    Left.at<float>(i, j) += cv::pow(disToY[i].x, degree - j);
                }
                Right.at<float>(i, 0) = disToY[i].y;
            }
            cv::solve(Left, Right, ABC, cv::DECOMP_SVD);
        }

        /**
         * 三维坐标转欧拉角工具函数
         * @param pts 三维坐标
         * @param pYaw
         * @param pPitch
         */
        static void convertPts2Euler(cv::Point3d &pts, float *pYaw, float *pPitch) {
            float _pitch = cv::fastAtan2(pts.y, cv::sqrt(pts.x * pts.x + pts.z * pts.z));
            float _yaw = cv::fastAtan2(pts.x, cv::sqrt(pts.y * pts.y + pts.z * pts.z));
            _pitch = _pitch > 180 ? _pitch - 360 : _pitch;
            *pPitch = -_pitch;
            *pYaw = _yaw > 180 ? _yaw - 360 : _yaw;
        }

        /**
         * 弹道修正函数
         * @param pts 原始坐标值
         * @param newPts 修正后坐标值
         */
        void correctTrajectory(cv::Point3d &pts, cv::Point3d &newPts) {
            if (isTest) {
                newPts.x = pts.x + armor::stConfig.get<int>("curve.test-dx");
                newPts.y = pts.y + armor::stConfig.get<int>("curve.test-dy");
                newPts.z = pts.z;
//                std::cout << "shoot z:" << pts.z << std::endl;
            } else {
                if (isLinear) {
                    /* 使用折线拟合 */
                    /* 使用二次形拟合 */
                    newPts.x = pts.x + m_stShootLines_x.getDelta(pts.z);
                    newPts.y = pts.y + m_stShootLines_y.getDelta(pts.z);
                    newPts.z = pts.z;
                } else {
                    /* 使用二次形拟合 */
                    newPts.x = pts.x + m_ABC_x.at<float>(0, 0) * pts.z * pts.z +
                               m_ABC_x.at<float>(1, 0) * pts.z + m_ABC_x.at<float>(2, 0);
                    newPts.y = pts.y + m_ABC_y.at<float>(0, 0) * pts.z * pts.z +
                               m_ABC_y.at<float>(1, 0) * pts.z + m_ABC_y.at<float>(2, 0);
                    newPts.z = pts.z;
                }
            }
        }
    } stCamera("../data/camera6mm.xml");


    // TODO: 测量, 实际检测灯长度不是55mm
    /**
     * 装甲板物理参数
     */
    struct {
        // 仅灯条矩形
        std::vector<cv::Point3d> smallFig3f = {
                cv::Point3d(0, 0, 0), cv::Point3d(0, -55, 0), cv::Point3d(135, -55, 0), cv::Point3d(135, 0, 0)
        };
        std::vector<cv::Point3d> largeFig3f = {
                cv::Point3d(0, 0, 0), cv::Point3d(0, -55, 0), cv::Point3d(230, -55, 0), cv::Point3d(230, 0, 0)
        };
        // 扩展区域
        std::vector<cv::Point2f> smallFig_Ex = {
                cv::Point2f(0, 0), cv::Point2f(0, 126), cv::Point2f(135, 126), cv::Point2f(135, 0)
        };
        std::vector<cv::Point2f> largeFig_Ex = {
                cv::Point2f(0, 0), cv::Point2f(0, 126), cv::Point2f(230, 126), cv::Point2f(230, 0)
        };
        cv::Mat smallShootPosition = cv::Mat(cv::Point3d(67.5, -27.5, 0.0));
    } stArmorStdFigure;

    /**
     * 灯条结构体
     */
    struct Light {
        cv::Point2f topPt;
        cv::Point2f bottomPt;
        cv::Point2f centerPt;
        double angle = 0;
        double length = 0;
    };

    typedef enum {
        TARGET_SMALL,
        TARGET_LARGE
    } emTargetType;

    /**
     * 目标结构体
     */
    struct Target {
        std::vector<cv::Point2f> pixelPts2f;  // 硬件ROI图幅下的像素坐标
        cv::Point2f pixelCenterPt2f;  // 像素坐标中心
        std::vector<cv::Point2f> pixelPts2f_Ex;  // 扩展像素坐标
        cv::Mat rv, tv, rvMat;
        cv::Point3d ptsInGimbal;  // 物体在云台坐标系下坐标(相机坐标系经过固定变换后得到)
        cv::Point3d ptsInWorld;  // 物体在世界坐标系下坐标
        cv::Point3d ptsInWorld_Predict;  // 物体在预测后的世界坐标系下坐标, 不开启预测的时候和 ptsInWorld 一样
        cv::Point3d ptsInGimbal_Predict;  // 物体在预测后的云台坐标系下坐标, 不开启预测的时候和 ptsInGimbal 一样
        cv::Point3d ptsInShoot;  // 物体在经过弹道修正后的云台坐标系下坐标
        float rPitch;  // 相对Pitch值, 发给电控
        float rYaw;  // 相对Yaw值, 发给电控
        int rTick;  // 相对帧编号
        emTargetType type;  // TARGET_SMALL, TARGET_TARGET

        cv::Mat m_rotY, m_rotX;  // 旋转到绝对坐标系
        cv::Point3d vInGimbal3d;

        explicit Target() : rPitch(0), rYaw(0), rTick(0), type(TARGET_SMALL) {}

        /**
         * 设置硬件ROI图幅下的像素坐标
         * @param a 左上
         * @param b 左下
         * @param c 右下
         * @param d 右上
         * @param startPt 开小图模式下的偏移量
         */
        void setPixelPts(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c, const cv::Point2f &d,
                         const cv::Point2i &startPt) {
            cv::Point2f startPt2f = cv::Point2f(startPt);
            pixelPts2f = std::vector<cv::Point2f>{a + startPt2f, b + startPt2f, c + startPt2f, d + startPt2f};
            for (const auto &pt:pixelPts2f) {
                pixelCenterPt2f += pt;
            }
            pixelCenterPt2f /= int(pixelPts2f.size());
        }

        /**
         * 灯条扩展
         * @return false = 扩展后超过硬件ROI图幅大小
         */
        bool convert2ExternalPts2f() {
            pixelPts2f_Ex.clear();
            cv::Point2f halfDeltaA = (pixelPts2f[0] - pixelPts2f[1]) / 55 * 35;
            pixelPts2f_Ex.emplace_back(pixelPts2f[0] + halfDeltaA);
            pixelPts2f_Ex.emplace_back(pixelPts2f[1] - halfDeltaA);
            cv::Point2f halfDeltaB = (pixelPts2f[3] - pixelPts2f[2]) / 55 * 35;
            pixelPts2f_Ex.emplace_back(pixelPts2f[2] - halfDeltaB);
            pixelPts2f_Ex.emplace_back(pixelPts2f[3] + halfDeltaB);

            for (const auto &_pt: pixelPts2f_Ex) {
                if (_pt.x >= stFrameInfo.size.width || _pt.x < 0 || _pt.y >= stFrameInfo.size.height || _pt.y < 0)
                    return false;
            }
            return true;
        }

        /**
         * 计算世界坐标参数
         */
        void calcWorldParams() {
            DEBUG("solvePnPRansac")
            /* 转化成相对原始图幅大小的像素坐标 */
            //PRINT_INFO("calcWorldParams started ");

            std::vector<cv::Point2d> gPixelPts2f;
            gPixelPts2f.resize(4);
            for (int i = 0; i < 4; ++i) {
                gPixelPts2f[i] = cv::Point2d(pixelPts2f[i]) + cv::Point2d(stFrameInfo.offset);
            }
//            PRINT_INFO("solvePnP before ");
//            std::cout<<"stArmorStdFigure.smallFig3f : "<<stArmorStdFigure.smallFig3f<<std::endl;
//            std::cout<<"gPixelPts2f  : "<<gPixelPts2f<<std::endl;
//            std::cout<<"stCamera.camMat : "<<stCamera.camMat<<std::endl;
//            std::cout<<"stCamera.distCoeffs :  "<<stCamera.distCoeffs<<std::endl;
//            std::cout<<"rv:  "<<rv << " tv : "<<tv <<std::endl;
            CV_Assert(!stCamera.camMat.empty());

            if (type == TARGET_SMALL)
                cv::solvePnP(stArmorStdFigure.smallFig3f, gPixelPts2f, stCamera.camMat, stCamera.distCoeffs, rv, tv);
            else
                cv::solvePnP(stArmorStdFigure.largeFig3f, gPixelPts2f, stCamera.camMat, stCamera.distCoeffs, rv, tv);
            //PRINT_INFO("solvePnP started ");

            cv::Rodrigues(rv, rvMat);


            /* TODO: 相机坐标系变换到云台坐标系(mm) */
            cv::Mat ptsInCamera_Mat = rvMat * stArmorStdFigure.smallShootPosition + tv;
            //PRINT_INFO("Rodrigues started ");

            DEBUG("ptsInCamera_Mat")
            ptsInGimbal.x = ptsInCamera_Mat.at<double>(0, 0);
            ptsInGimbal.y = ptsInCamera_Mat.at<double>(0, 1);
            ptsInGimbal.z = ptsInCamera_Mat.at<double>(0, 2) + 150;
            DEBUG("calcWorldParams end")
        }


        /**
         * 放弃
         * @param gYaw_
         * @param gPitch_
         */
        void convert2WorldPts(float gYaw_, float gPitch_) {
            gYaw_ = gYaw_ * M_PI / (180.0);
            gPitch_ = gPitch_ * M_PI / (180.0);

            /* yaw 为绕y轴旋转的 */
            m_rotY = (cv::Mat_<double>(3, 3)
                    <<
                    std::cos(gYaw_), 0, std::sin(gYaw_),
                    0, 1, 0,
                    -std::sin(gYaw_), 0, std::cos(gYaw_));

            /* pitch 为绕x轴旋转的 */
            m_rotX = (cv::Mat_<double>(3, 3)
                    <<
                    1, 0, 0,
                    0, std::cos(gPitch_), -std::sin(gPitch_),
                    0, std::sin(gPitch_), std::cos(gPitch_));

            /* 先绕动系y轴旋转, 再绕动系x轴旋转 */
            cv::Mat _pts = (cv::Mat_<double>(3, 1) << ptsInGimbal.x, ptsInGimbal.y, ptsInGimbal.z);
            cv::Mat ptsInWorldMat = m_rotY * m_rotX * _pts;
            ptsInWorld.x = ptsInWorldMat.at<double>(0);
            ptsInWorld.y = ptsInWorldMat.at<double>(1);
            ptsInWorld.z = ptsInWorldMat.at<double>(2);

            DEBUG("convert2WorldPts end")
        }

        void convert2GimbalPts(cv::Point3d &v) {
            cv::Mat _pts = (cv::Mat_<double>(3, 1) << ptsInWorld_Predict.x, ptsInWorld_Predict.y, ptsInWorld_Predict.z);
            cv::Mat m_rotY_inv = m_rotY.inv();
            cv::Mat m_rotX_inv = m_rotX.inv();
            cv::Mat _v_Mat = (cv::Mat_<double>(3, 1) << v.x, v.y, v.z);
            cv::Mat  ptsInGimbal_PredictMat = m_rotY_inv * m_rotX_inv * _pts;
            cv::Mat vInGimbal =  m_rotY_inv * m_rotX_inv * _v_Mat;
            vInGimbal3d.x = vInGimbal.at<double>(0);
            vInGimbal3d.y = vInGimbal.at<double>(1);
            vInGimbal3d.z = vInGimbal.at<double>(2);
            ptsInGimbal_Predict.x = ptsInGimbal_PredictMat.at<double>(0);
            ptsInGimbal_Predict.y = ptsInGimbal_PredictMat.at<double>(1);
            ptsInGimbal_Predict.z = ptsInGimbal_PredictMat.at<double>(2);
        }

        /**
         * 修正弹道 + 计算欧拉角
         */
        void correctTrajectory_and_calcEuler() {
            /* 弹道修正, TODO */
            stCamera.correctTrajectory(ptsInGimbal, ptsInShoot);
            DEBUG("stCamera.correctTrajectory")
            /* 计算欧拉角 */
            Camera::convertPts2Euler(ptsInShoot, &rYaw, &rPitch);
            DEBUG("Camera::convertPts2Euler")

        }
    };  // end struct Target

    class PID {
    private:
        double m_p, m_i, m_d;
        double m_sum;
        double lastError;
        int64_t lastTimeStamp;
        std::atomic_bool isFirst;
        std::mutex m_mutex;

        double limit(double in, double low, double up) { return in > up ? up : (in < low ? low : in); }

    public:

        explicit PID() : m_p(0), m_i(0), m_d(0), m_sum(0), lastError(0), lastTimeStamp(0), isFirst(true) {}

        /**
         *
         * @param p
         * @param i
         * @param d
         */
        void init(double p, double i, double d) {
            m_p = (p);
            m_i = (i);
            m_d = (d);
        }

        /**
         *
         * @param error
         * @param dt ms
         * @return
         */
        double calc(double error, int64_t timeStamp) {
            std::lock_guard<std::mutex> lockGuard(m_mutex);
            double out = error;
            if (!isFirst) {
                if (lastError * error < 0) m_sum = 0;
                m_sum += error;
                m_sum = limit(m_sum, -500, 500);
                double dt = (timeStamp - lastTimeStamp) / 1000000.0;
                dt = limit(dt, 0, 0.010);
                assert(dt > 0);
                out = error * m_p + m_sum * m_i * dt + (error - lastError) * m_d / dt;
            } else {
                isFirst = false;
                m_sum = 0;
            }
            lastError = error;
            lastTimeStamp = timeStamp;
            return limit(out, -5, 5);
        }

        double clear() {
            isFirst.exchange(true);
        }


    };

    /**
     * 放弃使用
     * 卡尔曼滤波线性预测
     * 状态量: [x, y, z, Delta_x, Delta_y, Delta_z]'
     * 观测量: [x, y, z]
     * 非线程安全
     */
    class Kalman {
        cv::KalmanFilter m_kf;
        cv::Mat m_measurement = cv::Mat::zeros(3, 1, CV_32F);
        int64_t m_lastTimeStamp;

    public:
        cv::Point3d velocity;

        explicit Kalman() : m_lastTimeStamp(0) {
            m_kf.init(6, 3, 0);
            m_kf.transitionMatrix = (cv::Mat_<float>(6, 6)
                    <<
                    1, 0, 0, 1, 0, 0,
                    0, 1, 0, 0, 1, 0,
                    0, 0, 1, 0, 0, 1,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 1);
            setIdentity(m_kf.measurementMatrix);
            setIdentity(m_kf.processNoiseCov, cv::Scalar::all(1e-5));
            setIdentity(m_kf.measurementNoiseCov, cv::Scalar::all(1e-1));
            setIdentity(m_kf.errorCovPost, cv::Scalar::all(1));
        }

        void clear_and_init(cv::Point3d &pos, int64_t timeStamp) {
            m_kf.transitionMatrix = (cv::Mat_<float>(6, 6)
                    <<
                    1, 0, 0, 1, 0, 0,
                    0, 1, 0, 0, 1, 0,
                    0, 0, 1, 0, 0, 1,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 1);
            setIdentity(m_kf.measurementMatrix);
            setIdentity(m_kf.processNoiseCov, cv::Scalar::all(1e-5));
            setIdentity(m_kf.measurementNoiseCov, cv::Scalar::all(1e-1));
            setIdentity(m_kf.errorCovPost, cv::Scalar::all(1));

            m_kf.statePost = (cv::Mat_<float>(6, 1) << pos.x, pos.y, pos.z, 0, 0, 0);
            m_lastTimeStamp = timeStamp;
        }

        /**
         * 本次数据的时间戳
         * @param pos 三维坐标
         * @param timeStamp 微秒
         */
        void correct(cv::Point3d &pos, int64_t timeStamp) {
            /* 计算时间差 */
            float deltaT = (timeStamp - m_lastTimeStamp) / 10000.0;  // s
            assert(deltaT > 0);
            m_lastTimeStamp = timeStamp;

            /* 更新状态转移矩阵 */
            m_kf.transitionMatrix = (cv::Mat_<float>(6, 6)
                    <<
                    1, 0, 0, deltaT, 0, 0,
                    0, 1, 0, 0, deltaT, 0,
                    0, 0, 1, 0, 0, deltaT,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 1);
            m_measurement.at<float>(0) = (float) pos.x;
            m_measurement.at<float>(1) = (float) pos.y;
            m_measurement.at<float>(2) = (float) pos.z;
            m_kf.correct(m_measurement);
        }

        /**
         * 预测值
         * @param delay 秒
         */
        void predict(float delay, cv::Point3d &predictRelativePos) {
            cv::Mat prediction = m_kf.predict();
            velocity = cv::Point3d(prediction.at<float>(3), prediction.at<float>(4), prediction.at<float>(5));
            predictRelativePos.x = prediction.at<float>(0) + delay * prediction.at<float>(3);
            predictRelativePos.y = prediction.at<float>(1) + delay * prediction.at<float>(4);
            predictRelativePos.z = prediction.at<float>(2) + delay * prediction.at<float>(5);
        }
    };
}

#endif //ATTACK_BASE_HPP
