#define _WIN32_WINNT 0x0602
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_EXPORT __declspec(dllexport)

typedef struct qwen_win32_direct_file {
    HANDLE file;
    HANDLE event;
    DWORD logical_sector;
    DWORD physical_sector;
    DWORD alignment;
    volatile LONG active;
    volatile LONG calls;
    unsigned long long bytes;
    double seconds;
    DWORD last_error;
    LARGE_INTEGER qpc_freq;
} qwen_win32_direct_file;

typedef struct qwen_win32_buffer {
    void *raw;
    void *aligned;
    unsigned long long bytes;
    unsigned long long allocation_bytes;
    DWORD alignment;
} qwen_win32_buffer;

static int qwen_is_power_of_two_u32(DWORD x) {
    return x != 0 && (x & (x - 1u)) == 0;
}

static DWORD qwen_max_u32(DWORD a, DWORD b) {
    return a > b ? a : b;
}

static wchar_t *qwen_utf8_to_wide(const char *path) {
    if (!path) return NULL;
    int n = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *wide = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!wide) return NULL;
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, n) <= 0) {
        free(wide);
        return NULL;
    }
    return wide;
}

QWEN_EXPORT void *qwen_win32_direct_open_utf8(const char *path_utf8) {
    wchar_t *path = qwen_utf8_to_wide(path_utf8);
    if (!path) return NULL;

    HANDLE file = CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL
            | FILE_FLAG_NO_BUFFERING
            | FILE_FLAG_OVERLAPPED
            | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL);
    free(path);
    if (file == INVALID_HANDLE_VALUE) return NULL;

    FILE_STORAGE_INFO info;
    memset(&info, 0, sizeof(info));
    if (!GetFileInformationByHandleEx(
            file, FileStorageInfo, &info, sizeof(info))) {
        CloseHandle(file);
        return NULL;
    }

    DWORD logical = info.LogicalBytesPerSector;
    DWORD physical = logical;
    physical = qwen_max_u32(physical, info.PhysicalBytesPerSectorForAtomicity);
    physical = qwen_max_u32(physical, info.PhysicalBytesPerSectorForPerformance);
    physical = qwen_max_u32(
        physical, info.FileSystemEffectivePhysicalBytesPerSectorForAtomicity);
    if (!qwen_is_power_of_two_u32(logical)
            || !qwen_is_power_of_two_u32(physical)) {
        CloseHandle(file);
        return NULL;
    }

    HANDLE event = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!event) {
        CloseHandle(file);
        return NULL;
    }

    qwen_win32_direct_file *ctx =
        (qwen_win32_direct_file *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        CloseHandle(event);
        CloseHandle(file);
        return NULL;
    }
    ctx->file = file;
    ctx->event = event;
    ctx->logical_sector = logical;
    ctx->physical_sector = physical;
    ctx->alignment = qwen_max_u32(logical, physical);
    ctx->active = 0;
    ctx->calls = 0;
    ctx->bytes = 0;
    ctx->seconds = 0.0;
    ctx->last_error = ERROR_SUCCESS;
    if (!QueryPerformanceFrequency(&ctx->qpc_freq)
            || ctx->qpc_freq.QuadPart <= 0) {
        CloseHandle(event);
        CloseHandle(file);
        free(ctx);
        return NULL;
    }
    return ctx;
}

QWEN_EXPORT int qwen_win32_direct_close(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    if (!ctx) return -1;
    if (InterlockedCompareExchange(&ctx->active, 0, 0) != 0) return -2;
    BOOL event_ok = CloseHandle(ctx->event);
    BOOL file_ok = CloseHandle(ctx->file);
    free(ctx);
    return (event_ok && file_ok) ? 0 : -3;
}

QWEN_EXPORT unsigned int qwen_win32_direct_logical_sector(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    return ctx ? (unsigned int)ctx->logical_sector : 0u;
}

QWEN_EXPORT unsigned int qwen_win32_direct_physical_sector(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    return ctx ? (unsigned int)ctx->physical_sector : 0u;
}

QWEN_EXPORT unsigned int qwen_win32_direct_alignment(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    return ctx ? (unsigned int)ctx->alignment : 0u;
}

QWEN_EXPORT unsigned int qwen_win32_direct_last_error(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    return ctx ? (unsigned int)ctx->last_error : (unsigned int)ERROR_INVALID_HANDLE;
}

QWEN_EXPORT unsigned long long qwen_win32_direct_calls(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    return ctx ? (unsigned long long)InterlockedCompareExchange(
        &ctx->calls, 0, 0) : 0ull;
}

QWEN_EXPORT unsigned long long qwen_win32_direct_bytes(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    return ctx ? ctx->bytes : 0ull;
}

QWEN_EXPORT double qwen_win32_direct_seconds(void *opaque) {
    qwen_win32_direct_file *ctx = (qwen_win32_direct_file *)opaque;
    return ctx ? ctx->seconds : 0.0;
}

QWEN_EXPORT int qwen_win32_direct_no_buffering(void *opaque) {
    return opaque ? 1 : 0;
}

QWEN_EXPORT int qwen_win32_direct_overlapped(void *opaque) {
    return opaque ? 1 : 0;
}

QWEN_EXPORT void *qwen_win32_buffer_create(
    unsigned long long bytes, unsigned int alignment_u32) {
    if (bytes == 0 || alignment_u32 == 0
            || !qwen_is_power_of_two_u32((DWORD)alignment_u32)) {
        return NULL;
    }
    unsigned long long extra = (unsigned long long)alignment_u32 - 1ull;
    if (bytes > UINT64_MAX - extra) return NULL;
    unsigned long long total = bytes + extra;
    if (total > (unsigned long long)SIZE_MAX) return NULL;

    void *raw = VirtualAlloc(
        NULL, (SIZE_T)total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!raw) return NULL;

    uintptr_t base = (uintptr_t)raw;
    uintptr_t mask = (uintptr_t)alignment_u32 - 1u;
    uintptr_t aligned_addr = (base + mask) & ~mask;
    if ((unsigned long long)(aligned_addr - base) + bytes > total) {
        VirtualFree(raw, 0, MEM_RELEASE);
        return NULL;
    }

    qwen_win32_buffer *buf =
        (qwen_win32_buffer *)calloc(1, sizeof(*buf));
    if (!buf) {
        VirtualFree(raw, 0, MEM_RELEASE);
        return NULL;
    }
    buf->raw = raw;
    buf->aligned = (void *)aligned_addr;
    buf->bytes = bytes;
    buf->allocation_bytes = total;
    buf->alignment = (DWORD)alignment_u32;
    return buf;
}

QWEN_EXPORT void *qwen_win32_buffer_ptr(void *opaque) {
    qwen_win32_buffer *buf = (qwen_win32_buffer *)opaque;
    return buf ? buf->aligned : NULL;
}

QWEN_EXPORT unsigned long long qwen_win32_buffer_bytes(void *opaque) {
    qwen_win32_buffer *buf = (qwen_win32_buffer *)opaque;
    return buf ? buf->bytes : 0ull;
}

QWEN_EXPORT unsigned int qwen_win32_buffer_alignment(void *opaque) {
    qwen_win32_buffer *buf = (qwen_win32_buffer *)opaque;
    return buf ? (unsigned int)buf->alignment : 0u;
}

QWEN_EXPORT int qwen_win32_buffer_destroy(void *opaque) {
    qwen_win32_buffer *buf = (qwen_win32_buffer *)opaque;
    if (!buf) return -1;
    BOOL ok = VirtualFree(buf->raw, 0, MEM_RELEASE);
    free(buf);
    return ok ? 0 : -2;
}

/*
 * Return codes:
 *   0  success
 *  -1  null/invalid argument
 *  -2  alignment contract violation
 *  -3  concurrent read attempted on single-I/O context
 *  -4  ReadFile failed before or outside ERROR_IO_PENDING
 *  -5  GetOverlappedResult failed
 *  -6  short read
 */
QWEN_EXPORT int qwen_win32_direct_read(
    void *file_opaque,
    void *buffer_opaque,
    unsigned long long buffer_offset,
    unsigned long long file_offset,
    unsigned int nbytes) {
    qwen_win32_direct_file *ctx =
        (qwen_win32_direct_file *)file_opaque;
    qwen_win32_buffer *buf = (qwen_win32_buffer *)buffer_opaque;
    if (!ctx || !buf || nbytes == 0) return -1;
    if (buffer_offset > buf->bytes
            || (unsigned long long)nbytes > buf->bytes - buffer_offset) {
        return -1;
    }

    unsigned char *dst =
        (unsigned char *)buf->aligned + (size_t)buffer_offset;
    if ((file_offset % ctx->logical_sector) != 0
            || ((unsigned long long)nbytes % ctx->logical_sector) != 0
            || ((uintptr_t)dst % ctx->alignment) != 0) {
        return -2;
    }
    if (InterlockedCompareExchange(&ctx->active, 1, 0) != 0) {
        return -3;
    }

    int rc = 0;
    ctx->last_error = ERROR_SUCCESS;
    ResetEvent(ctx->event);

    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)(file_offset & 0xffffffffull);
    ov.OffsetHigh = (DWORD)((file_offset >> 32) & 0xffffffffull);
    ov.hEvent = ctx->event;

    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    BOOL started = ReadFile(
        ctx->file, dst, (DWORD)nbytes, NULL, &ov);
    if (!started) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            ctx->last_error = err;
            rc = -4;
            goto done;
        }
    }

    DWORD got = 0;
    if (!GetOverlappedResult(ctx->file, &ov, &got, TRUE)) {
        ctx->last_error = GetLastError();
        rc = -5;
        goto done;
    }
    if (got != (DWORD)nbytes) {
        ctx->last_error = ERROR_HANDLE_EOF;
        rc = -6;
        goto done;
    }

    InterlockedIncrement(&ctx->calls);
    ctx->bytes += (unsigned long long)got;

done:
    QueryPerformanceCounter(&t1);
    ctx->seconds +=
        (double)(t1.QuadPart - t0.QuadPart)
        / (double)ctx->qpc_freq.QuadPart;
    InterlockedExchange(&ctx->active, 0);
    return rc;
}
