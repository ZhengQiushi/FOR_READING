#include"ThreadControl.hpp"
#include<thread>

int main()
{
    ThreadControl ThreadController;
    std::thread auto_aim(&ThreadControl::AutoAim, &ThreadController);
    std::thread communicate(&ThreadControl::Communicate, &ThreadController);
    
    auto_aim.join();
    communicate.join();
    return 0;
}