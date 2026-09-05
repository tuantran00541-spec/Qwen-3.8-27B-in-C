#ifndef QWEN38_THREAD_COMPAT_H
#define QWEN38_THREAD_COMPAT_H

/* Minimal persistent-worker synchronization abstraction for Qwen3.8 research.
 *
 * This deliberately exposes only the primitives needed by the exact Q6 row
 * pool.  Scheduler policy remains in the caller: static disjoint output-row
 * partitions, one main compute participant, and sleeping persistent workers.
 */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef HANDLE qwen_thread_t;
typedef SRWLOCK qwen_mutex_t;
typedef CONDITION_VARIABLE qwen_cond_t;
typedef DWORD (WINAPI *qwen_thread_fn)(LPVOID);

#define QWEN_THREAD_RET DWORD WINAPI
#define QWEN_THREAD_RETURN return 0
#define QWEN_EXPORT __declspec(dllexport)

static int qwen_mutex_init(qwen_mutex_t *m) {
    InitializeSRWLock(m);
    return 0;
}
static void qwen_mutex_destroy(qwen_mutex_t *m) { (void)m; }
static void qwen_mutex_lock(qwen_mutex_t *m) { AcquireSRWLockExclusive(m); }
static void qwen_mutex_unlock(qwen_mutex_t *m) { ReleaseSRWLockExclusive(m); }

static int qwen_cond_init(qwen_cond_t *c) {
    InitializeConditionVariable(c);
    return 0;
}
static void qwen_cond_destroy(qwen_cond_t *c) { (void)c; }
static int qwen_cond_wait(qwen_cond_t *c, qwen_mutex_t *m) {
    return SleepConditionVariableSRW(c, m, INFINITE, 0) ? 0 : -1;
}
static void qwen_cond_signal(qwen_cond_t *c) { WakeConditionVariable(c); }
static void qwen_cond_broadcast(qwen_cond_t *c) { WakeAllConditionVariable(c); }

static int qwen_thread_create(qwen_thread_t *out, qwen_thread_fn fn, void *arg) {
    HANDLE h = CreateThread(NULL, 0, fn, arg, 0, NULL);
    if (!h) return -1;
    *out = h;
    return 0;
}
static int qwen_thread_join(qwen_thread_t thread) {
    if (!thread) return -1;
    const DWORD rc = WaitForSingleObject(thread, INFINITE);
    const BOOL closed = CloseHandle(thread);
    return (rc == WAIT_OBJECT_0 && closed) ? 0 : -1;
}

static double qwen_wall_seconds(void) {
    LARGE_INTEGER counter, freq;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&freq);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

#else

#include <pthread.h>
#include <time.h>

typedef pthread_t qwen_thread_t;
typedef pthread_mutex_t qwen_mutex_t;
typedef pthread_cond_t qwen_cond_t;
typedef void *(*qwen_thread_fn)(void *);

#define QWEN_THREAD_RET void *
#define QWEN_THREAD_RETURN return NULL
#if defined(__GNUC__) || defined(__clang__)
#define QWEN_EXPORT __attribute__((visibility("default")))
#else
#define QWEN_EXPORT
#endif

static int qwen_mutex_init(qwen_mutex_t *m) { return pthread_mutex_init(m, NULL); }
static void qwen_mutex_destroy(qwen_mutex_t *m) { (void)pthread_mutex_destroy(m); }
static void qwen_mutex_lock(qwen_mutex_t *m) { (void)pthread_mutex_lock(m); }
static void qwen_mutex_unlock(qwen_mutex_t *m) { (void)pthread_mutex_unlock(m); }

static int qwen_cond_init(qwen_cond_t *c) { return pthread_cond_init(c, NULL); }
static void qwen_cond_destroy(qwen_cond_t *c) { (void)pthread_cond_destroy(c); }
static int qwen_cond_wait(qwen_cond_t *c, qwen_mutex_t *m) {
    return pthread_cond_wait(c, m);
}
static void qwen_cond_signal(qwen_cond_t *c) { (void)pthread_cond_signal(c); }
static void qwen_cond_broadcast(qwen_cond_t *c) { (void)pthread_cond_broadcast(c); }

static int qwen_thread_create(qwen_thread_t *out, qwen_thread_fn fn, void *arg) {
    return pthread_create(out, NULL, fn, arg);
}
static int qwen_thread_join(qwen_thread_t thread) { return pthread_join(thread, NULL); }

static double qwen_wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif

#endif
