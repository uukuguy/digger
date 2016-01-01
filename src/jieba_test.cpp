// jieba_test.cpp
//


#include "Jieba.hpp"

#include <sys/types.h>
#include <dirent.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


const char *jieba_dict = "../jieba/dict/jieba.dict.utf8";
const char *hmm_model = "../jieba/dict/hmm_model.utf8";
const char *user_dict = "../jieba/dict/user.dict.utf8";

//cppjieba::Jieba *jieba = NULL;
cppjieba::PosTagger *jieba = NULL;

std::string do_seg(const char *buf){
    std::string strTxt = "";

    typedef std::vector<std::pair<std::string, std::string> > Tags;
    Tags tags;
    jieba->Tag(buf, tags);
    
    for ( Tags::iterator it = tags.begin() ; it != tags.end() ; it++ ){
        std::pair<std::string, std::string>& tag = *it;
        //if ( tag.second == "n" || tag.second == "v" || tag.second == "vn" ) {
            //std::cout << tag << " ";
            strTxt += tag.first + ":" + tag.second + " ";
        //}
    }
    //std::cout << std::endl;

    return strTxt;
}

int do_file(const std::string &content_dir, const std::string &corpus_dir, const std::string &d_name)
{
    std::fstream file;

    std::string content_filename = content_dir + "/" + d_name;
    std::string corpus_filename = corpus_dir + "/" + d_name;

    file.open(content_filename.c_str());

    file.seekg(0, std::ios::end);
    uint32_t file_len = file.tellg();
    file.seekg(0, std::ios::beg);
    //std::cout << content_filename << "(" << file_len << ")" << std::endl;

    char *buf = new char[file_len+1];
    file.read(buf, file_len);
    buf[file_len] = 0;

    file.close();

    std::string strTxt = do_seg(buf);
    delete buf;

    std::ofstream corpus_file(corpus_filename.c_str(), std::ios::out | std::ios::trunc);
    corpus_file.write(strTxt.c_str(), strTxt.length());
    corpus_file.close();

    return 0;
}

int main(int argc, char *argv[])
{
    if ( argc < 3 ){
        std::cout << "Usage: " << argv[0] << " <content_dir> <corpus_dir>" << std::endl;
        exit(-1);
    }

    //jieba = new cppjieba::Jieba(jieba_dict, hmm_model, user_dict);
    jieba = new cppjieba::PosTagger(jieba_dict, hmm_model, user_dict);

    DIR *pDir;
    struct dirent *ent;
    const char *content_dir = argv[1];
    const char *corpus_dir = argv[2];
    uint32_t i = 0;

    pDir = opendir(content_dir);

    while ( (ent = readdir(pDir)) != NULL ){
        if ( ent->d_type & 8){
            do_file(content_dir, corpus_dir, ent->d_name);
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
