/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SMARTREDIS_THREADPOOL_H
#define SMARTREDIS_THREADPOOL_H

#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include <functional>

///@file

namespace SmartRedis {

class SRObject;

/*!
*   \brief  A thread pool for concurrent execution of parallel jobs
*/
class ThreadPool
{
  public:
    /*!
    *   \brief ThreadPool constructor
    *   \param context The owning context
    *   \param num_threads The number of threads to create in the pool,
    *          or 0 to use one thread per hardware context
    */
    ThreadPool(const SRObject* context, unsigned int num_threads=0);

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

    /*!
    *   \brief Owning client object
    */
    const SRObject* _context;
};

} // namespace SmartRedis
#endif // SMARTREDIS_THREADPOOL_H
