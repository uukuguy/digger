// jieba_test.cpp
//


#include "Jieba.hpp"

#include <sys/types.h>
#include <dirent.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


cppjieba::Jieba *jieba = NULL;

int do_seg(const char *buf){
    //std::cout << buf << std::endl;

    std::vector<std::string> words;
    jieba->Cut(buf, words, true);
    //std::cout << limonp::join(words.begin(), words.end(), "/") << std::endl;


    return 0;
}

int do_file(const char *file_name)
{
    std::fstream file;

    file.open(file_name);//, std::ios::binary);

    file.seekg(0, std::ios::end);
    uint32_t file_len = file.tellg();
    file.seekg(0, std::ios::beg);
    //std::cout << file_name << "(" << file_len << ")" << std::endl;

    char *buf = new char[file_len+1];
    file.read(buf, file_len);
    buf[file_len] = 0;

    file.close();

    do_seg(buf);

    delete buf;
    return 0;
}

int main(int argc, char *argv[])
{
    jieba = new cppjieba::Jieba("/home/jwsu/apps/jieba/dict/jieba.dict.utf8",
            "/home/jwsu/apps/jieba/dict/hmm_model.utf8",
            "/home/jwsu/apps/jieba/dict/user.dict.utf8");

    DIR *pDir;
    struct dirent *ent;
    const char *root_dir = argv[1];
    uint32_t i = 0;

    pDir = opendir(root_dir);

    while ( (ent = readdir(pDir)) != NULL ){
        if ( ent->d_type & 8){
            std::string file_name = std::string(root_dir) + "/" + ent->d_name;
            do_file(file_name.c_str());
            if ( (i / 100) * 100 == i ){
                std::cout << "(" << i << ")" << std::endl;
            }
            i += 1;
        }
    }

    std::cout << "Total: " << i << std::endl;

    closedir(pDir);

    delete jieba;

    //const char *xls_file = argv[1];
    //const void *hXLS = NULL;
    //if ( freexl_open(xls_file, &hXLS) != FREEXL_OK ) {
        //std::cout << "freexl_open() failed. " << xls_file << std::endl;
        //exit(-1);
    //}
    //if ( freexl_select_active_worksheet(hXLS, 0) != FREEXL_OK ){
        //std::cout << "freexl_active_worksheet() failed. " << std::endl;
        //exit(-1);
    //}

    //uint32_t rows = 0;
    //uint16_t cols = 0;
    //freexl_worksheet_dimensions(hXLS, &rows, &cols);

    //std::cout << "rows: " << rows << " cols: " << cols  << std::endl;

    //freexl_close(hXLS);
    
    return 0;
}
