#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include <functional>

#ifndef SMARTREDIS_THREADPOOL_H
namespace SmartRedis {

/*!
*   \brief  A thread pool for concurrent execution of parallel jobs
*/
class ThreadPool
{
  public:
    /*!
    *   \brief ThreadPool constructor
    *   \param num_threads The number of threads to create in the pool,
    *          or 0 to use one thread per hardware context
    */
    ThreadPool(unsigned int num_threads=0);

    /*!
    *   \brief ThreadPool destructor
    */
    ~ThreadPool();

    /*!
    *   \brief Shut down the thread pool. Blocks until all threads are terminated
    */
    void shutdown();

    /*!
    *   \brief Worker thread main loop to acquire and perform jobs
    *   \param tid Thread ID for the current thread
    */
    void perform_jobs(unsigned int tid);

    /*!
    *   \brief Submit a job to threadpool for execution
    *   \param job The job to be executed
    */
    void submit_job(std::function<void()> job);

  protected:
    /*!
    *   \brief The threads in our worker pool
    */
    std::vector<std::thread> threads;

    /*!
    *   \brief The current task queue of jobs waiting to be performed
    */
    std::queue<std::function<void()>> jobs;

    /*!
    *   \brief Lock, protecting the job queue
    */
    std::mutex queue_mutex;

    /*!
    *   \brief Condition variable for signalling worker threads
    */
    std::condition_variable cv;

    /*!
    *   \brief Flag if the thread pool initialization has completed
    */
    volatile bool initialization_complete;

    /*!
    *   \brief Flag for if thread pool shutdown has been triggered.
    */
    volatile bool shutting_down;

    /*!
    *   \brief Flag for if the thread pool shut down has completed.
    */
    volatile bool shutdown_complete;
};

} // namespace SmartRedis
#endif // SMARTREDIS_THREADPOOL_H
