#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "threadpool.h"
#include "srexception.h"

using namespace SmartRedis;
using namespace std::chrono_literals;

// TIME_THREADPOOL
// Enable this flag to display timings for the threadpool activity.
// Currently, this will be to screen, but eventually it will go to
// the SmartRedis log
#undef TIME_THREADPOOL

// Constructor
ThreadPool::ThreadPool(unsigned int num_threads)
{
    // Flag that we're initializing
    initialization_complete = false;

    // By default, we'll make one thread for each hardware context
    if (num_threads == 0)
        num_threads = std::thread::hardware_concurrency();

    // Create worker threads
	if (num_threads < 1) num_threads = 1; // Force a minimum of 1 thread
    for (unsigned int i = 0; i < num_threads; i++) {
        #ifdef TIME_THREADPOOL
        std::cout << "Kicking off thread " + std::to_string(i) << std::endl;
        #endif
        threads.push_back(std::thread(&ThreadPool::perform_jobs, this, i));
    }

    // Announce that we're open for business
    shutting_down = false;
    shutdown_complete = false;
    initialization_complete = true;
}

// Desstructor
ThreadPool::~ThreadPool()
{
    // Shut down the threadpool if it hasn't already been stopped
    if (!shutdown_complete) {
        shutdown();
    }
}

// Shut down the thread pool, blocking any further jobs from submission
void ThreadPool::shutdown()
{
    // Make sure initialization is complete before we shut down
    while (!initialization_complete)
        ; // Spin

    #ifdef TIME_THREADPOOL
    std::cout << "Shutting down thread pool" << std::endl;
    #endif

    // We're closed for business
    shutting_down = true;

    // Wait for worker threads to finish up
    #ifdef TIME_THREADPOOL
    int i = 0;
    size_t num_threads = threads.size();
    #endif
    for (std::thread& thr : threads) {
        cv.notify_all(); // Wake up all the threads
        #ifdef TIME_THREADPOOL
        std::cout << "Waiting for a thread to terminate (" << std::to_string(i++) << " of "
                  << std::to_string(num_threads) << ")" << std::endl;
        #endif
        thr.join(); // Blocks until the thread finishes execution
    }

    // Done
    #ifdef TIME_THREADPOOL
    std::cout << "Shutdown complete" << std::endl;
    #endif
    shutdown_complete = true;
}

// Worker thread main loop to acquire and perform jobs
void ThreadPool::perform_jobs(unsigned int tid)
{
    #ifdef TIME_THREADPOOL
    int jobid = 0;
    std::cout << "Thread " << std::to_string(tid) << " reporting for duty" << std::endl;
    #endif

    // Loop forever processing jobs until we get killed
    std::function<void()> job;
    while (!shutting_down)
    {
        #ifdef TIME_THREADPOOL
        auto start = std::chrono::steady_clock::now();
        #endif

        // Get a job, blocking until one appears if none immediately available
        do {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait_for(lock, 250ms, [this](){
                return !jobs.empty() || shutting_down;
            });

            // We woke up
            if (!shutting_down) {
                // Check for a spurious wakeup
                if (jobs.empty())
                    continue;
                // Otherwise, there's a job for us. Grab it
                job = jobs.front();
                jobs.pop();
                break;
            }
        } // End scope and release lock
        while (!shutting_down);

        #ifdef TIME_THREADPOOL
        auto have_job = std::chrono::steady_clock::now();
        #endif
        // Perform the job
        if (!shutting_down) {
            job();
            #ifdef TIME_THREADPOOL
            auto job_done = std::chrono::steady_clock::now();
            std::chrono::duration<double> get_job = have_job - start;
            std::chrono::duration<double> execute_job = job_done - have_job;
            std::cout << "Thread " << std::to_string(tid) << " "
                      << "time to get job " << std::to_string(jobid++) << ": " << get_job.count() << " s; "
                      << "time to execute job: " << execute_job.count() << " s" << std::endl;
            #endif
        }
    }
    #ifdef TIME_THREADPOOL
    std::cout << "Thread " << std::to_string(tid) << " shutting down" << std::endl;
    #endif
}

// Submit a job to threadpool for execution
void ThreadPool::submit_job(std::function<void()> job)
{
    // If the threadpool is shutting down, reject the job
    if (shutting_down) {
        throw SRInternalException(
            "Attempt made to add job to shutdown threadpool");
    }

    // Add the job to our queue
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.push(job);
    } // End scope and release lock

    // Wake up a sleeping worker to process it
    cv.notify_one();
}

