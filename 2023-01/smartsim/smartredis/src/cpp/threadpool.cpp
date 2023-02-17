#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "threadpool.h"
#include "srexception.h"
#include "logger.h"
#include "srobject.h"

using namespace SmartRedis;
using namespace std::chrono_literals;

// Constructor
ThreadPool::ThreadPool(const SRObject* context, unsigned int num_threads)
    : _context(context)
{
    // Flags that we're initializing and not shutting down
    initialization_complete = false;
    shutting_down = false;

    // By default, we'll make one thread for each hardware context
    if (num_threads == 0)
        num_threads = std::thread::hardware_concurrency();

    // Create worker threads
	if (num_threads < 1) num_threads = 1; // Force a minimum of 1 thread
    for (unsigned int i = 0; i < num_threads; i++) {
        _context->log_data(
            LLDeveloper, "Kicking off thread " + std::to_string(i));
        threads.push_back(std::thread(&ThreadPool::perform_jobs, this, i));
    }

    // Announce that we're open for business
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

    _context->log_data(LLDeveloper, "Shutting down thread pool");

    // We're closed for business
    shutting_down = true;

    // Wait for worker threads to finish up
    int i = 0;
    size_t num_threads = threads.size();
    for (std::thread& thr : threads) {
        cv.notify_all(); // Wake up all the threads
        std::string message =
            "Waiting for thread to terminate (" +
            std::to_string(i++) + " of " +
            std::to_string(num_threads) + ")";
        _context->log_data(LLDeveloper, message);
        thr.join(); // Blocks until the thread finishes execution
    }

    // Done
    _context->log_data(LLDeveloper, "Shutdown complete");
    shutdown_complete = true;
}

// Worker thread main loop to acquire and perform jobs
void ThreadPool::perform_jobs(unsigned int tid)
{
    int jobid = 0;
    _context->log_data(
        LLDebug, "Thread " + std::to_string(tid) + " reporting for duty");

    // Loop forever processing jobs until we get killed
    std::function<void()> job;
    while (!shutting_down)
    {
        auto start = std::chrono::steady_clock::now();

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

        auto have_job = std::chrono::steady_clock::now();

        // Perform the job
        if (!shutting_down) {
            job();
            auto job_done = std::chrono::steady_clock::now();
            std::chrono::duration<double> get_job = have_job - start;
            std::chrono::duration<double> execute_job = job_done - have_job;
            std::string message =
                "Thread " + std::to_string(tid) +
                " time to get job " + std::to_string(jobid++) +
                ": " + std::to_string(get_job.count()) + " s; " +
                "time to execute job: " +
                std::to_string(execute_job.count()) + " s";
            _context->log_data(LLDeveloper, message);
        }
    }

    _context->log_data(
        LLDeveloper, "Thread " + std::to_string(tid) + " shutting down");
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

