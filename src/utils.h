#ifndef __UTILS_H__
#define __UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> /* gettimeofday() struct timezone */
#include <time.h> /* struct tm, localtime() */

//#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
//#define roundup(x, y) ((((x) + ((y) - 1)) / (y)) * (y))

#if __BYTE_ORDER == __LITTLE_ENDIAN
#define __cpu_to_be16(x) bswap_16(x)
#define __cpu_to_be32(x) bswap_32(x)
#define __cpu_to_be64(x) bswap_64(x)
#define __be16_to_cpu(x) bswap_16(x)
#define __be32_to_cpu(x) bswap_32(x)
#define __be64_to_cpu(x) bswap_64(x)
#define __cpu_to_le32(x) (x)
#else
#define __cpu_to_be16(x) (x)
#define __cpu_to_be32(x) (x)
#define __cpu_to_be64(x) (x)
#define __be16_to_cpu(x) (x)
#define __be32_to_cpu(x) (x)
#define __be64_to_cpu(x) (x)
#define __cpu_to_le32(x) bswap_32(x)
#endif

static inline int before(uint32_t seq1, uint32_t seq2)
{
	return (int32_t)(seq1 - seq2) < 0;
}

static inline int after(uint32_t seq1, uint32_t seq2)
{
	return (int32_t)(seq2 - seq1) < 0;
}
/*
#define min(x, y) ({ \
        typeof(x) _x = (x);	\
        typeof(y) _y = (y);	\
        (void) (&_x == &_y);		\
        _x < _y ? _x : _y; })

#define max(x, y) ({ \
        typeof(x) _x = (x);	\
        typeof(y) _y = (y);	\
        (void) (&_x == &_y);		\
        _x > _y ? _x : _y; })
*/
/*#include <jemalloc/jemalloc.h>
static inline void *zalloc(size_t size)
{
    return calloc(1, size);
}
*/
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#define GET_TIME_MILLIS(msec) \
    uint64_t msec = 0; \
    { \
        struct timeval tv; \
        gettimeofday(&tv, NULL); \
        msec = tv.tv_sec * 1000 + tv.tv_usec / 1000; \
    }


extern int ones_in_binary(uint64_t lhs, uint64_t rhs, int n);

extern int get_instance_parent_full_path(char* apath, int size);

#ifdef __cplusplus
}
#endif

#endif
