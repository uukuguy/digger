/**
 * @file   filesystem.c
 * @author Jiangwen Su <uukuguy@gmail.com>
 * @date   2014-05-19 19:24:38
 * 
 * @brief  
 * 
 * 
 */

#include "filesystem.h"
#include "common.h"
#include "logger.h"

int mkdir_if_not_exist(const char *dirname)
{
    if ( access(dirname, F_OK) == -1 ){
        return mkdir(dirname, 0750);
    } 
    return 0;
}

int file_exist(const char *filename)
{
    if ( access(filename, F_OK) == 0 ){
        return 1;
    } else {
        return 0;
    }
}

int get_file_parent_full_path(const char *filename, char *apath, int size)
{
    int cnt = strlen(filename);

    if ( cnt > 0 ) {
        int i;
        for ( i = cnt ; i >= 0 ; i-- ){
            if ( filename[i] == '/' ) {
                memcpy(apath, filename, i);
                apath[i] = '\0';
                break;
            }
        }
        return i;
    }
    return 0;
}

int get_path_file_name(const char *path_name, char *file_name, int size)
{
    int cnt = strlen(path_name);

    int i;
    if ( cnt > 0 ){
        for ( i = cnt - 1 ; i > 0 ; i-- ){
            if ( path_name[i] == '/' ){
                int len = cnt - i - 1;
                memcpy(file_name, &path_name[i+1], len);
                file_name[len] = '\0';
                break;
            }
        }
        return i;
    }

    return 0;
}

