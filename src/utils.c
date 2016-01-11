/**
 * @file   utils.c
 * @author Jiangwen Su <uukuguy@gmail.com>
 * @date   2016-01-06 20:54:19
 * 
 * @brief  
 * 
 * 
 */

#include "logger.h"
#include "utils.h"

int ones_in_binary(uint64_t lhs, uint64_t rhs, int n)
{
    int cnt = 0;
    lhs ^= rhs;
    while ( lhs && cnt <= n ){
        lhs &= lhs - 1;
        cnt++;
    };
    if (cnt <= n ){
        return 0;
    } else {
        return -1;
    }
}

/**
 * apath must bin exe filename type: .../bin/foo
 */
int get_instance_parent_full_path(char* apath, int size)
{
    int cnt = 0;
#ifdef OS_LINUX
    cnt = readlink("/proc/self/exe", apath, size);
#endif

#ifdef OS_DARWIN
    realpath("./", apath);
    strcat(apath, "/bin/foo");
    cnt = strlen(apath);
#endif

    if ( cnt > 0 ) {
        int i;
        for ( i = cnt ; i >= 0 ; i-- ){
            if ( apath[i] == '/' ) {
                break;
            }
        }
        int j;
        for ( j = i - 1 ; j >= 0 ; j-- ){
            if ( apath[j] == '/' ) {
                apath[j] = '\0';
                break;
            }
        }
        return 0;
    } else {
        error_log("%d\n%d\n%s\n", cnt, errno, apath);
        return -1;
    }
}


