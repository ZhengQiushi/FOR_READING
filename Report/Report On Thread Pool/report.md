# 线程池实现研究报告
1950884 陆言

## 代码评注
> 其中 // 的部分为评注， /* */为原有注释。
```c++
class ThreadPool { // 对线程池的实现的研究
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
private:
    std::vector< std::thread > workers;/* 工作线程 */
    std::queue< std::function<void()> > tasks;/* 任务队列*/
    std::mutex queue_mutex;/* 保持队列同步的锁 */
    std::condition_variable condition;/* 结束线程条件 */
    bool stop;/* 线程池是否结束运行 */
};
 
/**
     * 构造函数
     * 在线程池内创建threads个线程
     * @param threads 线程数量
     */
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
/* 创建threads个工作线程 */
/* 所有工作线程都应被阻塞 */
// 下面大量使用了 C++11 的 lambda function ，对此作一定的注释：
// [函数对象参数] (操作符重载函数参数) mutable 或 exception 声明 -> 返回值类型 {函数体}
// 这里外面是一个类型为 void () 的 function ， this 代表可以和所在类共享变量。
// （但似乎这样几乎是把 lambda function 作为打包过程的工具，将过程打包为一个 anonymous function ，然后塞入 workers 这一队列）
// 作为省事的手段可以认同，但似乎对可读性并不纯是好处。
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                // 对于每个线程，用死循环来实现侦听消息队列。
                for(;;)
                {
                    std::function<void()> task;
                    {
                        // 给队列加读写锁。
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        
                        /* 谓语为假时阻塞线程 */
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        // 结束阻塞的条件是收到停止信号或者任务队列中有任务。
                        /* 防止出错，初始时stop=false，tasks为空 */
                        if(this->stop && this->tasks.empty())  // 收到结束信号且不再有任务。
                            return;    
                        // std::move 的意义在于将一个左值转化为右值引用。
                        // 左值：可以被赋值的值
                        // 右值：提供赋值的值
                        // 从队首取任务，并移出队列。 std::queue::front() 是一个 reference ，因而必须这样做。
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    // 执行任务。
                    task();
                }
            }
        );
}

// 构造函数中提及了任务队列的问题，这里就是实际将任务入队的地方。
/**
     * 任务入队
     */
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> // result_of 在 C++11 中被加入，在 C++20 中已经被移除，人啊就不知道，自己不可以预料。
    // 原本用于推导返回值。
    // CppReference: 类模板 std::future 提供访问异步操作结果的机制：
    //（通过 std::async 、 std::packaged_task 或 std::promise 创建的）异步操作能提供一个 std::future 对象给该异步操作的创建者。然后，异步操作的创建者能用各种方法查询、等待或从 std::future 提取值。若异步操作仍未提供值，则这些方法可能阻塞。异步操作准备好发送结果给创建者时，它能通过修改链接到创建者的 std::future 的共享状态（例如 std::promise::set_value ）进行。
{
    using return_type = typename std::result_of<F(Args...)>::type; // Sigh.
    
    /* 接受任务 */
    // CppReference: (std::make_shared) 构造 T 类型对象并将它包装于 std::shared_ptr 。
    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
    // std::bind: 调用此包装器等价于以一些绑定到 args 的参数调用 f 。它可以预先把指定可调用实体的某些参数绑定到已有的变量。
    // Currying procedure? 
    // std::forward: 转发左值为左值或右值，依赖于 T 。转发右值为右值并禁止右值的转发为左值。
    // std::packaged_task: 类模板 std::packaged_task 包装任何可调用 (Callable) 目标（函数、 lambda 表达式、 bind 表达式或其他函数对象），使得能异步调用它。
    // Generate an object of a function that can be called async-ly, which have been binded with certain params?
    // My Gosh.
    // It seems that C++ has ruin programmers' experience by making its type system too complex.
    
    // 从 得到一个 future 类型
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        /* 防止线程池结束工作时仍在入队 */
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        /* 任务入队 */
        // 又是个麻烦的 lambda function ，这里似乎也是因为 task 是一个 package_task ，不可能直接调用。用一个 lambda function 来，这里 () 是执行函数，则这个 lambda function 则在外面套了一层，使得一个 packaged_task 封装的函数又被封装成一个真实的函数 std::function<void()> 。 task.emplace 则是将这个 std::function<void()> 放入任务队列，等待空闲进程调用它。
        tasks.emplace([task](){ (*task)(); });
    }
    /* 随机唤醒一个等待的工作线程，开始工作 */
    condition.notify_one();

    return res;
}

/**
     * 析构函数
     * join所有线程
     */

// 线程池的析构函数。
inline ThreadPool::~ThreadPool()  
{
    // 保证没有人在读写任务队列。
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        // 修改全局状态变量，标志线程池 End Of Life 。
        stop = true;
    }
    
    // 唤醒进程池中所有在等待的进程，并让所有线程 join() ，生命周期结束。
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

```

## 调用过程分析
调用过程难以分析，主要问题在于为了处理各种类型之间的不兼容，这一段程序实在用了太多的 feature 。严谨调用过程分析又不得不分析这些类型之间的关系。此外，还有大量 callable object ，从逻辑上不是函数，但从形态上具有函数的特征。单从逻辑上，忽略一些细节大概可以这样叙述：    
`ThreadPools()` ： `std::queue<T>` , `std::condition_variable::wait()` , `std::move()` 。   
`enqueue()` ： `std::unique_lock` ， `std::queue<T>::emplaces()` ， `std::condition_variable::notify_one()` 。   
`~ThreadPool()` ： `std::unique_lock` , `std::condition_variable::notify_all()`, `std::thread::join()` 。   

## 优化
对不起，没有了.jpg