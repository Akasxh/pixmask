#include "common/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/cpu_caps.h"

namespace {

class ThreadPool {
public:
    ThreadPool();
    ~ThreadPool();

    void resize(std::size_t worker_count);
    std::size_t size() const noexcept;

    void enqueue(std::function<void()> task);
    void wait();

private:
    using Task = std::function<void()>;

    void start_workers(std::size_t worker_count);
    void stop_workers();
    void worker_loop(std::size_t index);
    bool try_pop_task(std::size_t index, Task& task);
    bool try_steal_task(std::size_t index, Task& task);
    void task_completed();

    std::vector<std::thread> threads_;
    std::vector<std::deque<Task>> queues_;
    std::vector<std::mutex> queue_mutexes_;

    mutable std::mutex work_mutex_;
    std::condition_variable work_cv_;
    std::condition_variable completion_cv_;

    std::atomic<bool> stop_{false};
    std::atomic<std::size_t> pending_tasks_{0};
    std::atomic<std::size_t> next_queue_{0};
};

ThreadPool::ThreadPool() {
    const std::size_t default_threads = std::max<std::size_t>(1, pixmask::hw_threads());
    start_workers(default_threads);
}

ThreadPool::~ThreadPool() {
    wait();
    stop_workers();
}

void ThreadPool::resize(std::size_t worker_count) {
    if (worker_count == 0) {
        worker_count = 1;
    }

    if (worker_count == threads_.size()) {
        return;
    }

    wait();
    stop_workers();
    start_workers(worker_count);
}

std::size_t ThreadPool::size() const noexcept {
    return threads_.size();
}

void ThreadPool::enqueue(std::function<void()> task) {
    if (!task) {
        return;
    }

    pending_tasks_.fetch_add(1, std::memory_order_relaxed);

    if (queues_.empty()) {
        task();
        task_completed();
        return;
    }

    const std::size_t queue_index = next_queue_.fetch_add(1, std::memory_order_relaxed) % std::max<std::size_t>(1, threads_.size());
    {
        std::lock_guard<std::mutex> lock(queue_mutexes_[queue_index]);
        queues_[queue_index].emplace_back(std::move(task));
    }

    std::lock_guard<std::mutex> lock(work_mutex_);
    work_cv_.notify_one();
}

void ThreadPool::wait() {
    if (pending_tasks_.load(std::memory_order_acquire) == 0) {
        return;
    }

    std::unique_lock<std::mutex> lock(work_mutex_);
    completion_cv_.wait(lock, [this]() { return pending_tasks_.load(std::memory_order_acquire) == 0; });
}

void ThreadPool::start_workers(std::size_t worker_count) {
    std::vector<std::deque<Task>> new_queues(worker_count);
    queues_.swap(new_queues);

    std::vector<std::mutex> new_mutexes(worker_count);
    queue_mutexes_.swap(new_mutexes);

    threads_.clear();
    threads_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        threads_.emplace_back([this, i]() { worker_loop(i); });
    }
}

void ThreadPool::stop_workers() {
    stop_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        work_cv_.notify_all();
    }

    for (auto& worker : threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    threads_.clear();
    queues_.clear();
    queue_mutexes_.clear();
    next_queue_.store(0, std::memory_order_relaxed);
    stop_.store(false, std::memory_order_release);
}

bool ThreadPool::try_pop_task(std::size_t index, Task& task) {
    if (index >= queues_.size()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(queue_mutexes_[index]);
    if (queues_[index].empty()) {
        return false;
    }

    task = std::move(queues_[index].front());
    queues_[index].pop_front();
    return true;
}

bool ThreadPool::try_steal_task(std::size_t index, Task& task) {
    const std::size_t worker_count = queues_.size();
    for (std::size_t offset = 1; offset < worker_count; ++offset) {
        const std::size_t target = (index + offset) % worker_count;
        std::lock_guard<std::mutex> lock(queue_mutexes_[target]);
        if (!queues_[target].empty()) {
            task = std::move(queues_[target].back());
            queues_[target].pop_back();
            return true;
        }
    }
    return false;
}

void ThreadPool::task_completed() {
    if (pending_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(work_mutex_);
        completion_cv_.notify_all();
        work_cv_.notify_all();
    }
}

void ThreadPool::worker_loop(std::size_t index) {
    while (true) {
        Task task;
        if (try_pop_task(index, task) || try_steal_task(index, task)) {
            task();
            task_completed();
            continue;
        }

        std::unique_lock<std::mutex> lock(work_mutex_);
        work_cv_.wait(lock, [this]() {
            return stop_.load(std::memory_order_acquire) || pending_tasks_.load(std::memory_order_acquire) > 0;
        });

        if (stop_.load(std::memory_order_acquire) && pending_tasks_.load(std::memory_order_acquire) == 0) {
            break;
        }
    }
}

ThreadPool& global_pool() {
    static ThreadPool pool;
    return pool;
}

} // namespace

namespace pixmask {

void set_threads(std::size_t worker_count) {
    global_pool().resize(worker_count);
}

std::size_t thread_count() noexcept {
    return global_pool().size();
}

void parallel_for_impl(
    std::size_t begin,
    std::size_t end,
    std::function<void(std::size_t, std::size_t)> chunk_fn) {
    if (!chunk_fn || end <= begin) {
        return;
    }

    ThreadPool& pool = global_pool();
    const std::size_t workers = std::max<std::size_t>(1, pool.size());
    const std::size_t total = end - begin;

    if (workers <= 1 || total <= workers) {
        chunk_fn(begin, end);
        return;
    }

    const std::size_t target_chunks = workers * 4;
    std::size_t chunk_size = (total + target_chunks - 1) / target_chunks;
    chunk_size = std::max<std::size_t>(1, chunk_size);

    std::size_t chunk_begin = begin;
    while (chunk_begin < end) {
        const std::size_t chunk_end = std::min(end, chunk_begin + chunk_size);
        pool.enqueue([chunk_fn, chunk_begin, chunk_end]() { chunk_fn(chunk_begin, chunk_end); });
        chunk_begin = chunk_end;
    }

    pool.wait();
}

} // namespace pixmask

extern "C" {

void pixmask_set_threads(std::size_t worker_count) {
    pixmask::set_threads(worker_count);
}

void pixmask_parallel_for(std::size_t begin,
                          std::size_t end,
                          void (*fn)(std::size_t, void*),
                          void* user_data) {
    if (!fn) {
        return;
    }

    pixmask::parallel_for(begin, end, [fn, user_data](std::size_t index) { fn(index, user_data); });
}

std::size_t pixmask_thread_count() {
    return pixmask::thread_count();
}

}
