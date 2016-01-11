#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <msgpack.hpp>
#include <tuple>
#include <shogun/base/init.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/machine/BaggingMachine.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/io/SerializableJsonFile.h>
#include <shogun/io/LibSVMFile.h>
using namespace shogun;

#include "Jieba.hpp"

#include <sys/types.h>
#include <stdarg.h>
#include <dirent.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <getopt.h>

#include "logger.h"

//#define NUM_UNLABELED_POSITIVE_SAMPLES 100
//#define NUM_UNLABELED_POSITIVE_SAMPLES 100
#define NUM_UNLABELED_POSITIVE_SAMPLES 20

//#define NUM_POSITIVE_SAMPLES 2000
//#define UNLABELED_RATIO 8.0
//#define UNLABELED_UNLABELED_RATIO 100.0

//#define NUM_POSITIVE_SAMPLES 1000
//#define UNLABELED_RATIO 3.0
//#define UNLABELED_UNLABELED_RATIO 200.0

#define NUM_POSITIVE_SAMPLES 3500
#define UNLABELED_RATIO 8.0
#define UNLABELED_UNLABELED_RATIO 0.0

const char* program_name = "udb";

// ==================== class Vocabulary ===================
class Vocabulary{

public:
    // std::tuple<std::string word, std::string pos>
    typedef std::tuple<std::string, std::string> WordTuple;
    static WordTuple make_word(const std::string& word, const std::string& pos){
        return std::make_tuple(word, pos);
    }

private:
    std::map<int, WordTuple> m_mapIdWords;
    std::map<std::string, int> m_mapWordIds;
    size_t m_max_term_id;
public:
    Vocabulary() 
    : m_max_term_id(0){
    }

    ~Vocabulary(){
    }

    // -------------------- clear() --------------------
    void clear(){
        m_mapIdWords.clear();
        m_mapWordIds.clear();
        m_max_term_id = 0;
    }

    // -------------------- add_word() --------------------
    int add_word(const std::string& word, const std::string& pos){
        int id = -1;
        std::map<std::string, int>::const_iterator it;
        it = m_mapWordIds.find(word);
        if ( it != m_mapWordIds.end() ){
            id = it->second;
        } else {
            id = m_max_term_id++;
            m_mapIdWords[id] = make_word(word, pos);
            m_mapWordIds[word] = id;
        }

        return id;
    }

    // -------------------- size() --------------------
    size_t get_max_term_id() const {
        return m_max_term_id;
    }

    // -------------------- has_word() --------------------
    bool has_word(int id) const{
        std::map<int, WordTuple>::const_iterator it;
        it = m_mapIdWords.find(id);
        if ( it != m_mapIdWords.end() ){
            return true;
        } else {
            return false;
        }
    }

    // -------------------- get_word_by_id() --------------------
    WordTuple get_word_by_id(int id) const{
        std::map<int, WordTuple>::const_iterator it;
        it = m_mapIdWords.find(id);
        if ( it != m_mapIdWords.end() ){
            return it->second;
        } else { 
            return make_word("", "");
        }
    }

    // -------------------- get_id_by_word() --------------------
    int get_id_by_word(const std::string& word) const {
        std::map<std::string, int>::const_iterator it;
        it = m_mapWordIds.find(word);
        if ( it != m_mapWordIds.end() ){
            return it->second;
        } else {
            return -1;
        }
    }

    // -------------------- diff() --------------------
    // from_vocabulary的词空间转换到to_vocabulary的词空间 
    // 返回from_id -> to_id 的转换表
    // std::tuple<from_id -> to_id, max_to_id)
    typedef std::tuple<std::map<int, int>, int> TransformTable;
    static TransformTable diff(const Vocabulary& from_vocabulary, const Vocabulary& to_vocabulary, bool add_not_exist = false)
    {
        std::map<int, int> idmap;

        int new_id = to_vocabulary.get_max_term_id();
        std::map<int, WordTuple>::const_iterator it;
        for ( it = from_vocabulary.m_mapIdWords.begin() ; it != from_vocabulary.m_mapIdWords.end() ; it++ ){
            int from_id = it->first;
            std::string word, pos;
            std::tie(word, pos) = it->second;
            int to_id = to_vocabulary.get_id_by_word(word);
            if ( to_id < 0 ){
                if ( add_not_exist ){
                    to_id = new_id++;
                    idmap.insert({from_id, to_id});
                }
            } else {
                idmap.insert({from_id, to_id});
            }
        }

        return std::make_tuple(idmap, new_id);
    }

    void transform(const TransformTable& transform_table)
    {
        std::map<int, int> idmap;
        int max_term_id;
        std::tie(idmap, max_term_id) = transform_table;

        std::map<int, WordTuple> mapIdWords;
        std::map<std::string, int> mapWordIds;

        for ( std::map<int, WordTuple>::const_iterator it = m_mapIdWords.begin() ; it != m_mapIdWords.end() ; it++ ){
            int term_id = it->first;
            std::map<int, int>::const_iterator it0 = idmap.find(term_id);
            if (it0 != idmap.end()){
                std::string word, pos;
                std::tie(word, pos) = it->second;
                int new_id = it0->second;
                mapIdWords.insert({new_id, std::make_tuple(word, pos)});
                mapWordIds.insert({word, new_id});
            }
        }

        m_max_term_id = max_term_id;

        m_mapIdWords = mapIdWords;
        m_mapWordIds = mapWordIds;
    }

    // -------------------- merge() --------------------
    std::map<int, int> merge(const Vocabulary& vocabulary){
        std::map<int, int> idmap;

        std::map<int, WordTuple>::const_iterator it;
        for ( it = vocabulary.m_mapIdWords.begin() ; it != vocabulary.m_mapIdWords.end() ; it++ ){
            int id = it->first;
            std::string word, pos;
            std::tie(word, pos) = it->second;
            int my_id = get_id_by_word(word);
            if ( my_id < 0 ){
               my_id = add_word(word, pos); 
            }
            idmap.insert({id, my_id});
        }

        return idmap;
    }

    // -------------------- save() --------------------
    int save(const std::string& filename) const{

        std::ofstream vocabulary_file(filename.c_str(), std::ios::out | std::ios::trunc);
        for ( std::map<int, WordTuple>::const_iterator it = m_mapIdWords.begin() ; it != m_mapIdWords.end() ; it++ ){
            //WordTuple w = it->second;
            std::string word, pos;
            std::tie(word, pos) = it->second;

            std::stringstream ss;
            //ss << it->first << " " << std::get<0>(w) << " " << std::get<1>(w) << std::endl;
            ss << it->first << " " << word << " " << pos << std::endl;
            std::string str = ss.str();
            vocabulary_file.write(str.c_str(), str.size());
        }
        vocabulary_file.close();

        return 0;
    }
}; // class Vocabulary

const char* jieba_root = "jieba";
const char* jieba_dict = "dict/jieba.dict.utf8";
const char* hmm_model = "dict/hmm_model.utf8";
const char* user_dict = "dict/user.dict.utf8";

//cppjieba::Jieba *jieba = NULL;
cppjieba::PosTagger *jieba = NULL;

extern "C"{
bool index_weight_compare(std::pair<int, double> first, std::pair<int, double> second){
    return first.second > second.second;
}
}

// ==================== class Samples ===================
class Samples{
public:
    typedef std::vector<double> Labels;

    // std::list<int term_id>
    typedef std::list<int> WordsList;
    // std::tuple<sample_id, label, ttile, words>
    typedef std::tuple<int, std::string, std::string, WordsList> SampleWords;
    typedef std::list<SampleWords> SampleWordsList;

    // -------------------- make_sample_words() --------------------
    static SampleWords make_sample_words(int sample_id, const std::string& label, const std::string& title, const WordsList& words){
        return std::make_tuple(sample_id, label, title, words);
    }

    // -------------------- print_sample_words() --------------------
    void print_sample_words(const SampleWords& sample_words) const
    {
        int sample_id;
        std::string label;
        std::string title;
        WordsList words;
        std::tie(sample_id, label, title, words) = sample_words;

        std::cout << "[" << sample_id << "] " << " - " << label << " - " << title << std::endl;
        for ( WordsList::const_iterator it = words.begin() ; it != words.end() ; it++ ){
            int term_id = *it;
            std::string word, pos;
            std::tie(word, pos) = m_vocabulary.get_word_by_id(term_id);
            std::cout << word << "(" << term_id << "," << pos << ")" << " ";
        }
        std::cout << std::endl;
    }

    // -------------------- print_sample_words_list() --------------------
    void print_sample_words_list(const SampleWordsList& sample_words_list) const
    {
        for ( SampleWordsList::const_iterator it = sample_words_list.begin() ; it != sample_words_list.end() ; it++ ){
            const Samples::SampleWords& sample_words = *it;
            print_sample_words(sample_words);
        }
    }

    void print_sample_words_list() const {
        print_sample_words_list(m_sample_words_list);
    }

    // std::map<int term_id, int term_used>
    typedef std::map<int, int> TermFrequencies;
    // std::tuple<int sample_id, int total_terms_used, TermFrequencies>
    typedef std::tuple<int, int, TermFrequencies> SampleTerms;
    typedef std::list<SampleTerms> SampleTermsList;

    // -------------------- make_sample_terms() --------------------
    static SampleTerms make_sample_terms(int sample_id, int total_terms_used, const TermFrequencies& term_frequencies)
    {
        return std::make_tuple(sample_id, total_terms_used, term_frequencies);
    }

    // -------------------- make_sample_terms() --------------------
    static SampleTerms make_sample_terms(int sample_id, const TermFrequencies& term_frequencies)
    {
        int total_terms_used = 0;
        for ( TermFrequencies::const_iterator it = term_frequencies.begin() ; it != term_frequencies.end() ; it++ ){
            int term_used = it->second;
            total_terms_used += term_used;
        }
        return std::make_tuple(sample_id, total_terms_used, term_frequencies);
    }

    // -------------------- print_sample_terms() --------------------
    void print_sample_terms(const SampleTerms& sample_terms) const
    {
        //typedef std::tuple<int, int, TermFrequencies> SampleTerms;
        //typedef std::map<int, int> TermFrequencies;
        int sample_id, total_terms_used;
        TermFrequencies term_frequencies;
        std::tie(sample_id, total_terms_used, term_frequencies) = sample_terms;
        std::cout << "[" << sample_id << "] " << total_terms_used << " ";

        for ( Samples::TermFrequencies::const_iterator it = term_frequencies.begin() ; it != term_frequencies.end() ; it++ ){
            int term_id = it->first;
            int term_used = it->second;
            std::string word, pos;
            std::tie(word, pos) = m_vocabulary.get_word_by_id(term_id);
            std::cout << word << "(" << term_id << "," << pos << ")" << ":" << term_used << " ";
        }

        std::cout << std::endl;
    }

    // -------------------- print_sample_terms_list() --------------------
    void print_sample_terms_list(const SampleTermsList& sample_terms_list) const
    {
        for ( SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++ ){
            const Samples::SampleTerms& sample_terms = *it;
            print_sample_terms(sample_terms);
        }
    }

    // -------------------- sample_words_to_sample_terms() --------------------
    static int sample_words_to_sample_terms(const SampleWordsList& sample_words_list, SampleTermsList& sample_terms_list) {
        for ( SampleWordsList::const_iterator it = sample_words_list.begin() ; it != sample_words_list.end() ; it++){
            int sample_id;
            //std::string title;
            WordsList words;
            std::tie(sample_id, std::ignore, std::ignore, words) = *it;

            TermFrequencies term_frequencies;
            for ( WordsList::const_iterator it0 = words.begin() ; it0 != words.end() ; it0++ ){
                int term_id = *it0;

                TermFrequencies::const_iterator it1 = term_frequencies.find(term_id);
                if ( it1 != term_frequencies.end() ){
                    term_frequencies[term_id] = it1->second + 1;
                } else {
                    term_frequencies.insert({term_id, 1});
                }
            }

            int total_terms_used = words.size();
            
            //SampleTerms sample_terms = make_sample_terms(sample_id, total_terms_used, term_frequencies);
            //sample_terms_list.push_back(sample_terms);

            sample_terms_list.emplace_back(sample_id, total_terms_used, term_frequencies);
        }

        debug_log("SampleWordsList -> SampleTermsList (%zu samples).", sample_terms_list.size());

        return 0;
    }

private:
    std::string m_samplesname;
    std::string m_samplesdir;
    Vocabulary m_vocabulary;
    SampleWordsList m_sample_words_list;
    Labels m_labels;

public:
    Samples(const std::string& samples_name)
    : m_samplesname(samples_name){
    }

    Samples(const std::string& samples_name, const std::string& samples_dir)
    : m_samplesname(samples_name), m_samplesdir(samples_dir){
    }

    ~Samples(){
    }

    std::string get_samples_name() const {
        return m_samplesname;
    }

    void set_samples_name(const std::string& samples_name){
        m_samplesname = samples_name;
    }

    size_t get_num_samples() const {
        return m_sample_words_list.size();
    }

    size_t get_num_features() const{
        return m_vocabulary.get_max_term_id();
    }

    const SampleWordsList& get_sample_words_list() const{
        return m_sample_words_list;
    }

    SampleWordsList& get_sample_words_list(){
        return m_sample_words_list;
    }

    const Labels& get_labels() const{
        return m_labels;
    }

    Labels& get_labels(){
        return m_labels;
    }

    void init_labels(double value){
        m_labels.clear();
        for ( int i = 0 ; i < get_num_samples() ; i++ ){
            m_labels.push_back(value);
        }
    }

    const Vocabulary& get_vocabulary() const{
        return m_vocabulary;
    }

    SGVector<float64_t>* get_sg_labels(){
        size_t num_samples = get_num_samples();
        size_t num_labels = m_labels.size();
        debug_log("num_labels = %zu num_samples = %zu", num_labels, num_samples);
        assert(num_samples == num_labels);

        SGVector<float64_t>* sglabels = new SGVector<float64_t>(num_labels);
        debug_log("sglabels.vlen = %d", sglabels->vlen);
        for ( int i = 0 ; i < num_labels ; i++ ){
            sglabels->set_element(m_labels[i], i);
        }
        return sglabels;
    }

    SGMatrix<float64_t>* to_sg_matrix(const SampleWordsList& sample_words_list) const {

        SampleTermsList sample_terms_list;
        sample_words_to_sample_terms(sample_words_list, sample_terms_list);
        //print_sample_terms_list(sample_terms_list);

        size_t num_features = get_num_features();
        size_t num_vectors = sample_terms_list.size();

        SGMatrix<float64_t>* mat = new SGMatrix<float64_t>(num_features, num_vectors);

        int idx = 0;
        for ( Samples::SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++, idx++ ){
            Samples::TermFrequencies term_frequencies;
            std::tie(std::ignore, std::ignore, term_frequencies) = *it;

            for ( Samples::TermFrequencies::const_iterator it0 = term_frequencies.begin() ; it0 != term_frequencies.end() ; it0++ ){
                int term_id = it0->first;
                int term_used = it0->second;
                (*mat)(term_id, idx) = term_used;
            }
        }
        debug_log("num_rows: %d  num_cols: %d", mat->num_rows, mat->num_cols);

        return mat;
    }

    SGMatrix<float64_t>* get_sg_matrix() const {
        return to_sg_matrix(m_sample_words_list);
    }

    int find_samples_has_term(const SampleTermsList& sample_terms_list, int term_id){
        int samples_has_term = 0;
        for ( SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++ ){
            TermFrequencies term_frequencies;
            std::tie(std::ignore, std::ignore, term_frequencies) = *it;
            TermFrequencies::const_iterator it0 = term_frequencies.find(term_id);
            if ( it0 != term_frequencies.end() ){
                samples_has_term++;
            }
        }

        return samples_has_term;
    }

    std::set<int> get_random_indices(size_t random_samples, size_t start_index = 0, size_t count=0) const{
        std::set<int> indices;

        srand((int)time(NULL));
        int x = get_num_samples();
        if (count >= random_samples ){
            x = count;
        }

        size_t cnt = 0;
        while ( cnt < random_samples ){
            int n = rand() % x;
            n += start_index;
            std::set<int>::iterator it = indices.find(n);
            if ( it != indices.end() ){
                continue;
            } else {
                indices.insert(std::set<int>::value_type(n));
            }
            if ( cnt % 1000 == 0 ){
                //debug_log("Get random indices: %zu/%zu", cnt, random_samples);
            }
            cnt++;
        }

        return indices;
    }

    Samples* subset(const std::string& samples_name, size_t count, size_t start_index = 0) const {
        notice_log("Subset %zu samples start with %zu", count, start_index);

        Samples* samples_clone = new Samples(samples_name);

        debug_log("Clone vocabulary.");
        samples_clone->m_vocabulary = m_vocabulary;

        debug_log("Clone sample_words_list and labels. (total %zu samples)", m_sample_words_list.size());
        int new_id = 0;
        int idx = 0;
        for ( SampleWordsList::const_iterator it = m_sample_words_list.begin() ; it != m_sample_words_list.end() ; it++, idx++ ){
            if ( idx < start_index ) {
                //debug_log("idx: %d < start_index: %zu count: %zu", idx, start_index, count);
                continue;
            }
            if ( idx >= start_index + count ) {
                debug_log("idx: %d >= start_index: %zu + %zu break!", idx, start_index, count);
                break;
            }

            int sample_id;
            std::string label;
            std::string title;
            WordsList words;
            std::tie(sample_id, label, title, words) = *it;

            if ( label.empty() ){
                std::stringstream ss;
                ss << get_samples_name() << "" << sample_id;
                label = ss.str();
            }

            samples_clone->m_sample_words_list.emplace_back(new_id++, label, title, words);
            samples_clone->m_labels.push_back(m_labels[idx]);
        }

        size_t total_samples = get_num_samples();
        size_t cloned_samples = samples_clone->get_num_samples();
        info_log("Subet %s %zu samples done (%zu samples).", m_samplesdir.c_str(),  count, cloned_samples);
        assert(cloned_samples == count || cloned_samples == total_samples - start_index);
        return samples_clone;
    }

    Samples* clone(const std::string& samples_name, size_t random_samples = 0, size_t start_index = 0, size_t count = 0) const {
        Samples* samples_clone = new Samples(samples_name);

        if ( random_samples == 0 ){
            samples_clone->m_vocabulary = m_vocabulary;
            samples_clone->m_sample_words_list = m_sample_words_list;
            samples_clone->m_labels = m_labels;
        } else if ( count == random_samples ) {
            debug_log("Clone vocabulary.");
            samples_clone->m_vocabulary = m_vocabulary;

            debug_log("Clone sample_words_list and labels. %zu samples in [%zu:%zu] (%zu samples)", random_samples, start_index, start_index + count - 1, count);
            int new_id = 0;
            int idx = 0;
            for ( SampleWordsList::const_iterator it = m_sample_words_list.begin() ; it != m_sample_words_list.end() ; it++, idx++ ){
                if (idx < start_index) continue;
                if ( idx >= start_index + count ) break;

                int sample_id;
                std::string label;
                std::string title;
                WordsList words;
                std::tie(sample_id, label, title, words) = *it;

                if ( label.empty() ){
                    std::stringstream ss;
                    ss << get_samples_name() << "" << sample_id;
                    label = ss.str();
                }

                samples_clone->m_sample_words_list.emplace_back(new_id++, label, title, words);
                samples_clone->m_labels.push_back(m_labels[idx]);
            }
        } else {
            debug_log("Clone vocabulary.");
            samples_clone->m_vocabulary = m_vocabulary;

            debug_log("Get random indices. %zu samples", random_samples);
            std::set<int> indices = get_random_indices(random_samples, start_index, count);

            debug_log("Clone sample_words_list and labels.");
            int new_id = 0;
            for ( std::set<int>::iterator it = indices.begin() ; it != indices.end() ; it++, new_id++){
                int n = *it;
                size_t idx = 0;
                SampleWordsList::const_iterator it0 =  m_sample_words_list.begin();
                while ( n-- > 0 ){
                    it0 ++;
                    idx++;
                }
                int sample_id;
                std::string label;
                std::string title;
                WordsList words;
                std::tie(sample_id, label, title, words) = *it0;
                if ( label.empty() ){
                    std::stringstream ss;
                    ss << get_samples_name() << "" << sample_id;
                    label = ss.str();
                }
                samples_clone->m_sample_words_list.emplace_back(new_id, label, title, words);
                samples_clone->m_labels.push_back(m_labels[idx]);
            }
        }

        info_log("Clone %s done (%zu samples).", m_samplesdir.c_str(), samples_clone->get_num_samples());
        return samples_clone;
    }

    // TODO
    SGSparseMatrix<float64_t>* to_tfidf_matrix(const SampleWordsList& sample_words_list, bool use_idf = true) const {
        SampleTermsList sample_terms_list;
        sample_words_to_sample_terms(sample_words_list, sample_terms_list);
        //print_sample_terms_list(sample_terms_list);

        int num_samples = get_num_samples();
        int num_features = get_num_features();

        SGSparseMatrix<float64_t>* tfidf_mat = new SGSparseMatrix<float64_t>(num_features, num_samples);
        SGSparseMatrix<float64_t>& mat = *tfidf_mat;

        float64_t sum_square_of_tfidf = 0.0;
        debug_log("tf");
        int row = 0;
        std::map<int, int> mapTermSamples;
        for ( SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++ , row++){
            int sample_id, total_terms_used;
            TermFrequencies term_frequencies;
            std::tie(sample_id, total_terms_used, term_frequencies) = *it;

            //if ( row % 1000 == 0 ){
                //debug_log("[tf] sample %d row %d", sample_id, row);
            //}

            for ( TermFrequencies::const_iterator it0 = term_frequencies.begin() ; it0 != term_frequencies.end() ; it0++ ){
                int term_id = it0->first;
                int term_used = it0->second;
                double tf = (double)term_used / (total_terms_used);
                //debug_log("term_used: %d total_terms_used: %d tf: %.6f", term_used, total_terms_used, tf);

                std::map<int, int>::iterator it1 = mapTermSamples.find(term_id);
                if ( it1 != mapTermSamples.end() ){
                    mapTermSamples[term_id] = mapTermSamples[term_id] + 1;
                } else {
                    mapTermSamples[term_id] = 1;
                }

                //debug_log("tf - row: %d  term_id: %d", row, term_id);
                mat(row, term_id) = tf;
            }
        }

        debug_log("use_idf");
        if ( use_idf ){
            int row = 0;
            for ( SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++ , row++){
                int sample_id, total_terms_used;
                TermFrequencies term_frequencies;
                std::tie(sample_id, total_terms_used, term_frequencies) = *it;

                //if ( row % 1000 == 0 ){
                    //debug_log("[idf] sample %d row %d", sample_id, row);
                //}
                for ( TermFrequencies::const_iterator it0 = term_frequencies.begin() ; it0 != term_frequencies.end() ; it0++ ){
                    int term_id = it0->first;
                    int samples_has_term = mapTermSamples[term_id];
                    double idf = log((double)num_samples / samples_has_term + 0.01);
                    //debug_log("idf - row: %d  term_id: %d", row, term_id);
                    float64_t tfidf = mat(row, term_id) * idf;
                    //debug_log("tfidf: %.6f", tfidf);
                    sum_square_of_tfidf += tfidf * tfidf;
                    mat(row, term_id) = tfidf;
                }
            }
        }

        debug_log("Do normalization");
        float64_t alpha = 1.0 / sqrt(sum_square_of_tfidf);

        //for ( size_t row = 0 ; row < mat.num_vectors ; row++ ){
            //for ( size_t col = 0 ; col < mat.num_features ; col++ ){
                //float64_t tfidf = mat(row, col);
                //if ( tfidf != 0.0 ){
                    //mat(row,col) = alpha * tfidf;
                //}
            //}
        //}

        row = 0;
        for ( SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++ , row++){
            TermFrequencies term_frequencies;
            std::tie(std::ignore, std::ignore, term_frequencies) = *it;
            for ( TermFrequencies::const_iterator it0 = term_frequencies.begin() ; it0 != term_frequencies.end() ; it0++ ){
                int term_id = it0->first;
                float64_t tfidf = mat(row, term_id);
                mat(row, term_id) = alpha * tfidf;
            }
        }

    debug_log("to_tfidf_matrix() done.");

        return tfidf_mat;
    }

    SGSparseMatrix<float64_t>* get_tfidf_matrix(bool use_idf = true) const{
        return to_tfidf_matrix(m_sample_words_list, use_idf);
    }

    int load(const std::string& samples_dir = ""){
        if ( !samples_dir.empty())
            m_samplesdir = samples_dir;
        info_log("Load %s", m_samplesdir.c_str());

        load_vocabulary();
        size_t num_features = get_num_features();
        info_log("%zu features loaded.", num_features);

        load_words();
        //print_sample_words_list(sample_words_list);

        // labels
        init_labels(1.0);

        return 0;
    }

    // -------------------- get_content_dir() --------------------
    std::string get_dbcontent_dir() const{
        std::string dbcontent_dir = m_samplesdir + "/content";
        return dbcontent_dir;
    }

    // -------------------- get_dbwords_dir() --------------------
    std::string get_dbwords_dir() const{
        std::string dbwords_dir = m_samplesdir + "/words";
        return dbwords_dir;
    }

    // -------------------- get_words_filename() --------------------
    std::string get_words_filename() const{
        std::string words_filename = m_samplesdir + "/words.txt";
        return words_filename;
    }

    // -------------------- get_vocabulary_dir() --------------------
    std::string get_dbvocabulary_dir() const{
        std::string dbvocabulary_dir = m_samplesdir + "/vocabulary";
        return dbvocabulary_dir;
    }

    // -------------------- get_vocabulary_filename() --------------------
    std::string get_vocabulary_filename() const{
        std::string vocabulary_filename = m_samplesdir + "/vocabulary.txt";
        return vocabulary_filename;
    }

    leveldb::DB* open_dbcontent() const{
        leveldb::DB* dbcontent;

        leveldb::Status status;
        leveldb::Options options;
        std::string dbcontent_dir = get_dbcontent_dir();
        status = leveldb::DB::Open(options, dbcontent_dir, &dbcontent);
        if ( !status.ok() ) {
            error_log("Open leveldb %s failed.", dbcontent_dir.c_str());
            error_log("%s", status.ToString().c_str());
            return NULL;
        }

        return dbcontent;
    }

    void close_dbcontent(leveldb::DB* dbcontent) const{
        delete dbcontent;
    }

    // -------------------- rebuild_words() --------------------
    // create samples_dir/words from samples_dir/content.
    int rebuild_words(){

        if ( jieba == NULL ){

            char root_dir[NAME_MAX];
            get_instance_parent_full_path(root_dir, NAME_MAX);
            std::string str_jieba_root = std::string(root_dir) + "/" + std::string(jieba_root); 
            std::string str_jieba_dict = str_jieba_root + "/" + std::string(jieba_dict);
            std::string str_hmm_model = str_jieba_root + "/" + std::string(hmm_model);
            std::string str_user_dict = str_jieba_root + "/" + std::string(user_dict);
            jieba = new cppjieba::PosTagger(str_jieba_dict, str_hmm_model, str_user_dict);
        }

        leveldb::Options options;
        leveldb::Status status;
        
        leveldb::DB* dbcontent = open_dbcontent();

        leveldb::DB* dbwords;
        options.create_if_missing = true;
        std::string dbwords_dir = get_dbwords_dir();
        status = leveldb::DB::Open(options, dbwords_dir, &dbwords);
        if ( !status.ok() ){
            error_log("Open leveldb %s failed.", dbwords_dir.c_str());
            error_log("%s", status.ToString().c_str());
            return -1;
        }
        leveldb::WriteBatch batchWords;

        std::string words_filename = get_words_filename();
        std::ofstream words_file(words_filename, std::ios::out | std::ios::trunc);

        m_vocabulary.clear();

        size_t cnt = 0;
        leveldb::Iterator* iter = dbcontent->NewIterator(leveldb::ReadOptions());
        iter->SeekToFirst();
        while ( iter->Valid() ){
            leveldb::Slice slice_key = iter->key();
            leveldb::Slice slice_value = iter->value();
            std::string key = slice_key.ToString();
            std::string value = slice_value.ToString();

            msgpack::unpacked result;
            msgpack::unpack(result, value.data(), value.size());
            msgpack::object deserialized = result.get();

            msgpack::type::tuple<int, std::string, std::string> dst;
            try{
                deserialized.convert(&dst);
            } catch (msgpack::type_error e){
                error_log("sample %s msgpack type mismatched.", key.c_str());
            }

            int sample_id;
            std::string title, content;
            msgpack::type::tie(sample_id, title, content) = dst;

            
            if ( cnt % 100 == 0 ){
                debug_log("sample %d: %s", sample_id, title.c_str());
            }
           
            std::stringstream ss;
            ss  << sample_id << " ";

            WordsList words = do_word_segmentation(title + "\n" + content);

            // TODO
            std::string str_sample_words = serialize_sample_words(sample_id, title, words);
            //std::stringstream ss1;
            //ss1 << sample_id;
            char buf[64];
            sprintf(buf, "%d", sample_id);
            batchWords.Put(std::string(buf), str_sample_words);

            for ( WordsList::const_iterator it = words.begin() ; it != words.end() ; it++ ){
                int term_id = *it;
                std::string word, pos;
                std::tie(word, pos) = m_vocabulary.get_word_by_id(term_id);

                ss << word << ":" << pos << " ";
            }
            ss << std::endl;
            std::string str = ss.str();
            words_file.write(str.c_str(), str.size());

            iter->Next();
            cnt++;
        };
        delete iter;

        std::string vocabulary_filename = get_vocabulary_filename();
        m_vocabulary.save(vocabulary_filename);

        words_file.close();

        dbwords->Write(leveldb::WriteOptions(), &batchWords);
        delete dbwords;

        delete dbcontent;

        delete jieba;
        jieba = NULL;

        return 0;
    }

    // -------------------- load_words() --------------------
    // load sample_words array from samples_dir/words.
    int load_words(){
        info_log("%s Loading words...", m_samplesdir.c_str());
        m_sample_words_list.clear();

        leveldb::DB* dbwords;
        std::string dbwords_dir = get_dbwords_dir();
        leveldb::Status status = leveldb::DB::Open(leveldb::Options(), dbwords_dir, &dbwords);
        if ( !status.ok() ){
            error_log("Open leveldb %s failed.", dbwords_dir.c_str());
            error_log("%s", status.ToString().c_str());
            return -1;
        }

        leveldb::Iterator* iter = dbwords->NewIterator(leveldb::ReadOptions());
        iter->SeekToFirst();
        while ( iter->Valid() ){
            leveldb::Slice slice_key = iter->key();
            leveldb::Slice slice_value = iter->value();

            std::string key = slice_key.ToString();
            std::string value = slice_value.ToString();

            SampleWords sample_words = deserialize_sample_words(value);
            int sample_id;
            std::tie(sample_id, std::ignore, std::ignore, std::ignore) = sample_words;
            int key_id = atoi(key.c_str());
            assert(sample_id == key_id);

            m_sample_words_list.emplace_back(sample_words);

            iter->Next();
        };
        delete iter;

        delete dbwords;

        size_t num_samples = get_num_samples();
        info_log("%s %zu samples loaded.", m_samplesdir.c_str(), num_samples);

        return 0;
    }

    // -------------------- load_vocabulary() --------------------
    int load_vocabulary(){

        info_log("Loading vocabulary...");

        m_vocabulary.clear();
        std::string vocabulary_filename = get_vocabulary_filename();
        std::ifstream vocabulary_file(vocabulary_filename);

        std::string line;
        while ( std::getline(vocabulary_file, line) ){
            std::stringstream ss(line);

            std::string str_term_id;
            std::getline(ss, str_term_id, ' ');
            //int term_id = atoi(str_term_id.c_str());

            std::string word;
            std::getline(ss, word, ' ');

            std::string pos;
            std::getline(ss, pos, ' ');

            m_vocabulary.add_word(word, pos);
        };

        return 0;
    }

    // 将本样本集的词空间更新为参数样本集的词空间
    void update_term_namespace(const Samples& master_samples){
    }

    // 同步两个样本集的词空间为原先两个词空间的并集
    // 并修正参数样本集samples中m_sample_words_list中的term_id。
    void sync_vocabulary(Samples& samples){
        std::map<int, int> idmap = m_vocabulary.merge(samples.m_vocabulary);
        samples.m_vocabulary = m_vocabulary;

        SampleWordsList sample_words_list = samples.m_sample_words_list;
        samples.m_sample_words_list.clear();

        SampleWordsList::const_iterator it;
        for ( it = sample_words_list.begin() ; it != sample_words_list.end() ; it++ ){
            int sample_id;
            WordsList words;
            std::string label;
            std::string title;
            std::tie(sample_id, label, title, words) = *it;

            for ( WordsList::iterator it0 = words.begin() ; it0 != words.end() ; it0++ ){
                int word = *it0;
                std::map<int, int>::const_iterator it_idmap = idmap.find(word);
                if ( it_idmap != idmap.end() ){
                    int word_new = it_idmap->second;
                    *it0 = word_new;
                } else {
                    error_log("Error: word (id:%d) does not exist.", word);
                    exit(-1);
                }
            }
            samples.m_sample_words_list.emplace_back(sample_id, label, title, words);
        }
    }

    void transform_vocabulary(const Vocabulary::TransformTable& transform_table){
        m_vocabulary.transform(transform_table);
        
        std::map<int, int> idmap;
        //int max_term_id;
        std::tie(idmap, std::ignore) = transform_table;

        SampleWordsList sample_words_list;

        SampleWordsList::const_iterator it;
        for ( it = m_sample_words_list.begin() ; it != m_sample_words_list.end() ; it++ ){
            int sample_id;
            WordsList words;
            std::string label;
            std::string title;
            std::tie(sample_id, label, title, words) = *it;

            WordsList new_words;
            for ( WordsList::iterator it0 = words.begin() ; it0 != words.end() ; it0++ ){
                int word = *it0;
                std::map<int, int>::const_iterator it_idmap = idmap.find(word);
                if ( it_idmap != idmap.end() ){
                    int word_new = it_idmap->second;
                    new_words.emplace_back(word_new); 
                } else {
                    continue;
                }
            }
            sample_words_list.emplace_back(sample_id, label, title, new_words);
        }

        m_sample_words_list = sample_words_list;
    }

    // 添加另一个样本集到本样本集中，本样本集的词空间扩充为原先两个词空间的并集。
    int append(const Samples& samples){
        std::map<int, int> idmap = m_vocabulary.merge(samples.m_vocabulary);

        int sample_id = get_num_samples();
        SampleWordsList::const_iterator it;
        for ( it = samples.m_sample_words_list.begin() ; it != samples.m_sample_words_list.end() ; it++ ){
            WordsList words;
            int s_id;
            std::string label;
            std::string title;
            std::tie(s_id, label, title, words) = *it;
            if ( label.empty() ){
                std::stringstream ss;
                ss << samples.get_samples_name() << "" << s_id;
                label = ss.str();
            }

            for ( WordsList::iterator it0 = words.begin() ; it0 != words.end() ; it0++ ){
                int word = *it0;
                std::map<int, int>::const_iterator it_idmap = idmap.find(word);
                if ( it_idmap != idmap.end() ){
                    int word_new = it_idmap->second;
                    *it0 = word_new;
                } else {
                    error_log("Error: word (id:%d) does not exist.", word);
                    exit(-1);
                }
            }

            m_sample_words_list.emplace_back(sample_id++, label, title, words);
        }

        for ( std::vector<double>::const_iterator it = samples.m_labels.begin() ; it != samples.m_labels.end() ; it++ ){
            m_labels.push_back(*it);
        }

        return 0;
    }

    int query_sample(int sample_id){
        leveldb::DB* dbwords;
        std::string dbwords_dir = get_dbwords_dir();
        leveldb::Status status = leveldb::DB::Open(leveldb::Options(), dbwords_dir, &dbwords);
        if ( !status.ok() ){
            error_log("Open leveldb %s failed.", dbwords_dir.c_str());
            error_log("%s", status.ToString().c_str());
            return -1;
        }

        std::stringstream ss;
        ss << sample_id;
        std::string strKey = ss.str();
        leveldb::Slice key(strKey);
        std::string value;
        status = dbwords->Get(leveldb::ReadOptions(), key, &value);
        if ( !status.ok() ){
            error_log("Read leveldb %s for sample %d key(%s) failed.", dbwords_dir.c_str(), sample_id, key.ToString().c_str());
            error_log("%s", status.ToString().c_str());
            delete dbwords;
            return -1;
        }
        delete dbwords;

        //std::string key = slice_key.ToString();
        //std::string value = slice_value.ToString();

        SampleWords sample_words = deserialize_sample_words(value);
        int s_id;
        std::string label;
        std::string title;
        WordsList words;
        std::tie(s_id, label, title, words) = sample_words;
        debug_log("query sample_id = %d result s_id = %d", sample_id, s_id);

        info_log("Sample(%d) %s %s", sample_id, label.c_str(), title.c_str());
        std::stringstream sswords;
        for ( WordsList::const_iterator it = words.begin() ; it != words.end() ; it++ ){
            int term_id = *it;
            std::string word, pos;
            std::tie(word, pos) = m_vocabulary.get_word_by_id(term_id);
            sswords << word << ":" << pos << " ";
        }
        //info_log("%zu words %s", words.size(), sswords.str().c_str());
        std::cout << words.size() << " words " << sswords.str() << std::endl;

        return 0;
    }

private:

    // -------------------- do_word_segmentation() --------------------
    WordsList do_word_segmentation(const std::string& content){
        WordsList words;

        const std::set<std::string> stopwords;/* = {"发表", "回复", "注册", "注册名", "帖子",
            "倒序", "作者", "浏览", "请问", "造成", "网址", "错误", "无法", "获取", "读取",
            "没有", "希望", "谢谢你", "用户", "女子", "工作人员", "栏杆", "二手车", "楼主",
            "网友", "人行道", "摘编", "复制", "公网", "传媒", "广播", "专属", "区域", "持有",
            "电视节目", "房主", "夫妇", "农民", "幼儿园", "孩子", "浏览器", "手机", "用户",
            "临时工", "版块", "店方", "父亲", "母亲", "村长", "高血压", "公墓", "陵园",
            "上传", "下载", "次数", "附件", "挖得", "转会费", "球员", "蒸饭", "餐馆",
            "青年宫", "足球队", "俱乐部", "高考", "名校", "谢谢", "本贴", "工资", 
            "居民", "楼顶", "通知", "奖金", "年终奖", "劳动", "劳动者", "放假",
            "投诉", "举报", "收费", "收费员", "农村", "领导", "员工", "小时", "男友", "女士"
        };*/

        typedef std::vector<std::pair<std::string, std::string> > Tags;
        Tags tags;
        jieba->Tag(content.c_str(), tags);
        
        for ( Tags::iterator it = tags.begin() ; it != tags.end() ; it++ ){
            std::pair<std::string, std::string>& tag = *it;
            std::string word = tag.first;
            std::string pos = tag.second;
            //debug_log("word: %s length: %d", word.c_str(), word.length());

            if ( word.length() <= 3 ) continue;
            if ( stopwords.find(word) != stopwords.end() ) continue;

            if ( (pos.substr(0, 1) == "n" && pos != "ns") || pos == "v" || pos == "vn" ) {
                int term_id = m_vocabulary.add_word(word, pos);    
                words.push_back(term_id);
            }
        }

        return words;
    }

    // -------------------- serialize_sample_words() --------------------
    std::string serialize_sample_words(int sample_id, const std::string& title, const WordsList& words) const{
        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> packer(&sbuf);

        packer.pack(sample_id);
        packer.pack(title);
        int num_words = words.size();
        packer.pack_array(num_words);
        for ( WordsList::const_iterator it = words.begin() ; it != words.end() ; it++ ){
            int term_id = *it;
            packer.pack(term_id);
        }

        return std::string(sbuf.data(), sbuf.size());
    }

    typedef std::tuple<int, std::string, std::string> SampleContent;
    SampleContent deserialize_sample_content(const std::string& buffer) const{
        SampleContent sample_content;

        msgpack::unpacked result;
        msgpack::unpack(result, buffer.data(), buffer.size());
        msgpack::object deserialized = result.get();

        msgpack::type::tuple<int, std::string, std::string> dst;
        try{
            deserialized.convert(&dst);
        } catch (msgpack::type_error e){
            error_log("sample msgpack type mismatched.");
        }

        int sample_id;
        std::string title, content;
        msgpack::type::tie(sample_id, title, content) = dst;

        sample_content = std::make_tuple(sample_id, title, content);

        return sample_content;
    }

    // -------------------- deserialize_sample_words() --------------------
    SampleWords deserialize_sample_words(const std::string& buffer) const{
        SampleWords sample_words;
        WordsList words;

        size_t buffer_size = buffer.size();

        msgpack::unpacker upk;
        upk.reserve_buffer(buffer_size);
        memcpy(upk.buffer(), buffer.data(), buffer_size);
        upk.buffer_consumed(buffer_size);

        //debug_log("buffer len = %zu", buffer.length());
        msgpack::unpacked result;
        if (!upk.next(&result)){
            error_log("No sample_id in msgpack buffer.");
            return -1;
        }
        msgpack::object objSampleId = result.get();
        if ( objSampleId.type != msgpack::type::POSITIVE_INTEGER ){
            error_log("Wrong format for sample_id.");
            return -1;
        }
        int sample_id = objSampleId.as<int>();

        if (!upk.next(&result)){
            error_log("No title in msgpack buffer.");
            return -1;
        }
        msgpack::object objTitle = result.get();
        if ( objTitle.type != msgpack::type::STR ){
            error_log("Wrong format for title.");
            return -1;
        }
        std::string title = objTitle.as<std::string>();

        if (!upk.next(&result)){
            error_log("No word list in msgpack buffer.");
            return -1;
        }
        msgpack::object objWords = result.get();
        if ( objWords.type != msgpack::type::ARRAY ){
            error_log("Wrong format for word list.");
            return -1;
        }
        //int num_items = objWords.via.array.size;
        if ( objWords.via.array.size > 0 ){
            msgpack::object* pi = objWords.via.array.ptr;
            msgpack::object* pi_end = objWords.via.array.ptr + objWords.via.map.size;
            do {
                int term_id = pi[0].as<int>();
                words.push_back(term_id);
                ++pi;
            } while (pi < pi_end);
        }

        sample_words = make_sample_words(sample_id, "", title, words);
        return sample_words;
    }

public:
    void print_pred_result(const std::vector<double>& pred_result) const{
        std::vector<std::pair<int, double> > a;
        for ( int i = 0 ; i < pred_result.size() ; i++ ){
            double v = pred_result[i];
            //if ( v <= 0.0 ) continue;
            a.emplace_back(i, v);
        }

        std::sort(a.begin(), a.end(), index_weight_compare);

        
        std::map<int, std::tuple<int, std::string, std::string> > mapSamples;
        size_t idx = 0;
        for ( SampleWordsList::const_iterator it = m_sample_words_list.begin() ; it != m_sample_words_list.end() ; it++, idx++ ){
            //double v = pred_result[idx];
            //if ( v <= 5.0 ) continue;
            int sample_id;
            std::string label;
            std::string title;
            std::tie(sample_id, label, title, std::ignore) = *it;

            mapSamples[idx] = std::make_tuple(sample_id, label, title);
            //debug_log("[%zu] Score: %.6f %s", idx, v, title.c_str());
            //print_sample_words(sample_words);
        }

        int num_recall_1 = 0;
        int num_recall_2 = 0;
        int num_recall_3 = 0;
        std::ofstream predict_result_file("./data/predict_result.txt", std::ios::out | std::ios::trunc);
        for (int i = 0 ; i < a.size() ; i++ ){
            std::pair<int,double> b = a[i];
            int idx = b.first;

            if ( idx < NUM_UNLABELED_POSITIVE_SAMPLES ){
                if ( i < NUM_UNLABELED_POSITIVE_SAMPLES ){
                    num_recall_1++;
                }
                if ( i < NUM_UNLABELED_POSITIVE_SAMPLES * 2){
                    num_recall_2++;
                }
                if ( i < NUM_UNLABELED_POSITIVE_SAMPLES * 3){
                    num_recall_3++;
                }
            }

            double v = b.second;
            int sample_id;
            std::string label;
            std::string title;
            std::tie(sample_id, label, title) = mapSamples[idx];

            //debug_log("[%d] Score: %.6f %s", idx, v, title.c_str());
            std::stringstream ss;
            ss << "[" << sample_id << "] " << " - " << label << " - " << "Score: " << v << " " << title << std::endl;
            std::string line = ss.str();
            predict_result_file.write(line.c_str(), line.length());
        }
        double recall_1 = (double)num_recall_1 / NUM_UNLABELED_POSITIVE_SAMPLES;
        double recall_2 = (double)num_recall_2 / NUM_UNLABELED_POSITIVE_SAMPLES;
        double recall_3 = (double)num_recall_3 / NUM_UNLABELED_POSITIVE_SAMPLES;
        notice_log("Recall: first - %d(%.3f%%)  second - %d(%.3f%%) third - %d(%.3f%%)", num_recall_1, recall_1, num_recall_2, recall_2, num_recall_3, recall_3);

        predict_result_file.close();
        notice_log("File ./data/predict_result.txt writed. %zu samples predicted.", a.size());


    }

}; // class Samples


float64_t predict_accuracy(CEvaluation* eval, CBinaryLabels* pred_labels, CBinaryLabels* labels, size_t num_positive_samples = 0)
{
    float64_t accuracy = eval->evaluate(pred_labels, labels);

    if ( num_positive_samples == 0 ){
        labels->get_int_labels().display_vector();
        pred_labels->get_int_labels().display_vector();
    } else {
        CBinaryLabels labels_100(num_positive_samples);
        CBinaryLabels pred_labels_100(num_positive_samples);

        for ( int i = 0 ; i < num_positive_samples ; i++ ){
            labels_100.set_label(i, labels->get_label(i));
            pred_labels_100.set_label(i, pred_labels->get_label(i));
        }
        //labels_100.get_int_labels().display_vector();
        //pred_labels_100.get_int_labels().display_vector();
        float64_t accuracy_100 = eval->evaluate(&pred_labels_100, &labels_100);


        int pred_positive_samples_100 = 0;
        //for ( size_t i = 0 ; i < pred_labels_100.get_num_labels() ; i++ ){
            //float64_t v = pred_labels_100.get_value(i);
            //if ( v > 0.0 ){
                //pred_positive_samples_100++;
            //}
        //}
        int pred_positive_samples = 0;
        for ( size_t i = 0 ; i < pred_labels->get_num_labels() ; i++ ){
            float64_t v = pred_labels->get_value(i);
            //debug_log("pred_positive_samples v[%zu]: %.6f", i, v);
            if ( v > 0.0 ){
                pred_positive_samples++;
                if ( i < num_positive_samples ){
                    pred_positive_samples_100++;
                }
            }
        }

        debug_log("accuracy_100: %.6f Precision: %d /  %d = %.6f", accuracy_100, pred_positive_samples_100, pred_positive_samples, 
                (double)pred_positive_samples_100  / pred_positive_samples);
    }

    return accuracy;
}

// std::tuple<accuracy, eval, labels>
typedef std::tuple<float64_t, CEvaluation*, CLabels*> TrainResult;

TrainResult train_and_eval(CMachine* machine, CFeatures* train_data, CBinaryLabels* train_labels)
{
    machine->train(train_data);

    CBinaryLabels* pred_labels = machine->apply_binary(train_data);
    //debug_log("machine->apply_binary(train_data)");
    //pred_labels->get_labels().display_vector();

    CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();
    float64_t accuracy = predict_accuracy(eval, pred_labels, train_labels, NUM_POSITIVE_SAMPLES);

    return std::make_tuple(accuracy, eval, static_cast<CLabels*>(pred_labels));
}

CMachine* build_liblinear_machine(CSparseFeatures<float64_t>* train_data, CBinaryLabels* train_labels)
{
    CLibLinear* liblinear_machine= new CLibLinear(2.0, train_data, train_labels);
    liblinear_machine->set_bias_enabled(true);
    float64_t C_neg = 10.0;
    float64_t C_pos = 1.0;
    liblinear_machine->set_C(C_neg, C_pos);

    float64_t accuracy;
    CEvaluation* eval;
    CBinaryLabels* pred_labels;
    TrainResult train_result = train_and_eval(liblinear_machine, train_data, train_labels);
    std::tie(accuracy, eval, (CLabels*&)pred_labels) = train_result;

    info_log("Train accuracy: %.3f", accuracy);

    SG_UNREF(eval);
    SG_UNREF(pred_labels);

    return liblinear_machine;
}

CMachine* build_bagging_machine(CSparseFeatures<float64_t>* train_data, CBinaryLabels* train_labels)
{
    int num_bags = 5;
    int bag_size = 25;
    CLibLinear* liblinear_machine = new CLibLinear();
    liblinear_machine->set_bias_enabled(true);
    float64_t C_neg = 50.0;
    float64_t C_pos = 1.0;
    liblinear_machine->set_C(C_neg, C_pos);

    liblinear_machine->set_bias_enabled(true);
    CMajorityVote* mv = new CMajorityVote();

    CBaggingMachine* bagging_machine = new CBaggingMachine(train_data, train_labels);
    bagging_machine->set_num_bags(num_bags);
    bagging_machine->set_bag_size(bag_size);
    bagging_machine->set_machine(liblinear_machine);
    bagging_machine->set_combination_rule(mv);

    TrainResult train_result = train_and_eval(bagging_machine, train_data, train_labels);

    float64_t accuracy;
    CEvaluation* eval;
    CBinaryLabels* pred_labels;
    std::tie(accuracy, eval, (CLabels*&)pred_labels) = train_result;

    float64_t oob_error = bagging_machine->get_oob_error(eval);

    debug_log("accuracy: %.3f  oob error: %.3f", accuracy, oob_error);

    SG_UNREF(eval);
    SG_UNREF(pred_labels);

    return bagging_machine;
}

int save_liblinear_machine(CLibLinear* liblinear_machine, const std::string& model_filename)
{
    SGVector<float64_t> w = liblinear_machine->get_w();
    float64_t bias = liblinear_machine->get_bias();

    info_log("Save liblinear machine to file %s", model_filename.c_str());
    info_log("bias = %.6f", bias);
    std::stringstream ss0;
    ss0 << "w.vlen = " << w.vlen << " w = [";
    //for (int i = 0 ; i < w.vlen ; i++ ){
        //float64_t v = w.get_element(i);
        //ss0 << v << ", ";
    //}
    ss0 << "]";

    std::cout << ss0.str() << std::endl;
    //info_log("w.vlen = %d w = [%s]", w.vlen, ss0.str().c_str());


    std::stringstream ss;
    ss << bias << std::endl;
    ss << ss0.str();
    std::string strContent = ss.str();

    std::ofstream model_file(model_filename.c_str(), std::ios::out | std::ios::trunc);
    model_file.write(strContent.c_str(), strContent.length());
    model_file.close();

    return 0;
}

void export_vocabulary_weights(const std::string& filename, const Vocabulary& vocabulary, const SGVector<float64_t>& w)
{
    std::vector<std::pair<int, double> > a;
    for ( int i = 0 ; i < w.vlen ; i++ ){
        float64_t weight = w[i];
        a.emplace_back(i, weight);
    }
    std::sort(a.begin(), a.end(), index_weight_compare);

    std::string strContent;
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::trunc);

    for ( int i = 0 ; i < a.size() ; i++ ){
        int id;
        double weight;
        std::tie(id, weight) = a[i];
        std::string word, pos;
        std::tie(word, pos) = vocabulary.get_word_by_id(id);

        std::stringstream ss;
        ss << word << "(" << id << ") " << weight << std::endl;
        std::string line = ss.str();
        file.write(line.c_str(), line.length());
    }

    file.close();
}
// TODO 
//std::vector<double> BiasedSVM(const SGSparseMatrix<float64_t>* train_data_mat, const SGVector<float64_t>* train_labels_vec, const SGSparseMatrix<float64_t>* test_data_mat, const SGVector<float64_t>* test_labels_vec){
std::tuple<CMachine*, CLabels*> BiasedSVM(const SGSparseMatrix<float64_t>* train_data_mat, const SGVector<float64_t>* train_labels_vec, const SGSparseMatrix<float64_t>* test_data_mat, const SGVector<float64_t>* test_labels_vec){

    CSparseFeatures<float64_t>* train_data = new CSparseFeatures<float64_t>(*train_data_mat);
    CBinaryLabels* train_labels = new CBinaryLabels(*train_labels_vec);
    debug_log("SGSparseMatrix<float64_t>* train_data_mat: vectors: %d features: %d", train_data_mat->num_vectors, train_data_mat->num_features);
    debug_log("SGVector<float64_t>* train_labels_vec: vlen: %d", train_labels_vec->vlen);
    debug_log("CSparseFeatures<float64_t>* train_data: features: %d", train_data->get_num_vectors());

    //CMachine* machine = build_bagging_machine(train_data, train_labels);

    CMachine* machine = build_liblinear_machine(train_data, train_labels);
    save_liblinear_machine(static_cast<CLibLinear*>(machine), "./data/pu.model");

    //machine->print_serializable("");

    //CSerializableAsciiFile asciiFile("test.model", 'w');
    //machine->save_serializable(&asciiFile);
    
    CSparseFeatures<float64_t>* test_data = new CSparseFeatures<float64_t>(*test_data_mat);
    CBinaryLabels* test_labels = new CBinaryLabels(*test_labels_vec);
    debug_log("SGSpareMatrix<float_t>* test_data_mat: vectors: %d features: %d", test_data_mat->num_vectors, test_data_mat->num_features);
    debug_log("SGVector<float64_t>* test_labels_vec: vlen: %d", test_labels_vec->vlen);
    debug_log("CSparseFeatures<float64_t>* test_data: features: %d", test_data->get_num_vectors());

    CBinaryLabels* pred_labels = machine->apply_binary(test_data);
    debug_log("machine->apply_binary(test_data)");
    //pred_labels->get_labels().display_vector();

    CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();
    float64_t accuracy = predict_accuracy(eval, pred_labels, test_labels, NUM_UNLABELED_POSITIVE_SAMPLES);
    info_log("Predict accuracy: %.3f", accuracy);

    SG_UNREF(eval);

    //SG_UNREF(pred_labels);

    //SG_UNREF(machine);

    //return pred_result;

    return std::make_tuple(machine, pred_labels);
}


void rebuild_words(Samples& positive_samples, Samples& unlabeled_samples, Samples& test_samples){
    positive_samples.rebuild_words();
    unlabeled_samples.rebuild_words();
    test_samples.rebuild_words();
}

void test(Samples& train_positive_samples, Samples& train_unlabeled_samples, Samples& test_positive_samples, Samples& test_unlabeled_samples)
{

    Samples train_samples = train_positive_samples;
    debug_log("Train positive samples cloned. %zu samples  %zu features.", train_samples.get_num_samples(), train_samples.get_num_features());

    train_samples.append(train_unlabeled_samples);
    debug_log("Train unlabeled samples appended. %zu samples  %zu features.", train_samples.get_num_samples(), train_samples.get_num_features());

    Samples test_samples = test_positive_samples;
    debug_log("Test positive samples cloned. %zu samples  %zu features.", test_samples.get_num_samples(), test_samples.get_num_features());

    test_samples.append(test_unlabeled_samples);
    debug_log("Test unlabeled samples appended. %zu samples  %zu features.", test_samples.get_num_samples(), test_samples.get_num_features());

    //test_samples.sync_vocabulary(train_samples);
    debug_log("transform vocabulary...");
    debug_log("Train samples %zu samples %zu features.", train_samples.get_num_samples(), train_samples.get_num_features());
    debug_log("Test samples %zu samples %zu features.", test_samples.get_num_samples(), test_samples.get_num_features());
    Vocabulary::TransformTable transform_table = Vocabulary::diff(test_samples.get_vocabulary(), train_samples.get_vocabulary(), false);
    test_samples.transform_vocabulary(transform_table);
    debug_log("Transform vocabulary Done.");
    debug_log("Test samples %zu samples %zu features.", test_samples.get_num_samples(), test_samples.get_num_features());

    notice_log("BiasedSVM()...");

    SGSparseMatrix<float64_t>* train_data_mat = train_samples.get_tfidf_matrix();
    SGVector<float64_t>* train_labels_vec = train_samples.get_sg_labels();

    //CLibSVMFile train_svmfile = CLibSVMFile("train.svm", 'w');
    //train_svmfile.set_sparse_matrix(train_data_mat->sparse_matrix, train_samples.get_num_features(), train_samples.get_num_samples());
    //train_svmfile.save_serializable((CSerializableFile*)&train_svmfile);

    SGSparseMatrix<float64_t>* test_data_mat = test_samples.get_tfidf_matrix();
    SGVector<float64_t>* test_labels_vec = test_samples.get_sg_labels();

    //CLibSVMFile test_svmfile = CLibSVMFile("test.svm", 'w');
    //test_svmfile.set_sparse_matrix(test_data_mat->sparse_matrix, test_samples.get_num_features(), test_samples.get_num_samples());
    //test_svmfile.save_serializable((CSerializableFile*)&test_svmfile);

    CMachine* machine;
    CLabels* pred_labels;
    std::tie(machine, pred_labels) = BiasedSVM(train_data_mat, train_labels_vec, test_data_mat, test_labels_vec);

    CLibLinear* liblinear_machine = static_cast<CLibLinear*>(machine);
    export_vocabulary_weights("./data/vocabulary_weights.txt", test_samples.get_vocabulary(), liblinear_machine->get_w());

    std::vector<double> pred_result;
    for ( size_t i = 0 ; i < pred_labels->get_num_labels() ; i++ ){
        float64_t v = pred_labels->get_value(i);
        pred_result.push_back(v);
    }
    test_samples.print_pred_result(pred_result);


    delete train_data_mat;
    delete train_labels_vec;
    delete test_data_mat;
    delete test_labels_vec;
}

void learning(Samples& positive_samples, Samples& unlabeled_samples, Samples& test_samples)
{
    positive_samples.load();
    positive_samples.init_labels(1.0);
    __attribute__((unused)) size_t total_positive_samples = positive_samples.get_num_samples();

    unlabeled_samples.load();
    unlabeled_samples.init_labels(-1.0);
    //size_t total_unlabeled_samples = unlabeled_samples.get_num_samples();

    // -----------------------------
    // positive.samples + unlabeled.samples to train
    // predict test.samples

    test_samples.load();
    test_samples.init_labels(-1.0);

    int num_positive_samples = NUM_POSITIVE_SAMPLES;
    int num_unlabeled_samples = (double)NUM_POSITIVE_SAMPLES * UNLABELED_RATIO;
    int num_unlabeled_positive_samples = NUM_UNLABELED_POSITIVE_SAMPLES;

    int random_samples = num_positive_samples + num_unlabeled_positive_samples;
    Samples* random_positive_samples = positive_samples.clone("rpos", random_samples, 0, random_samples);
    Samples* random_unlabeled_samples = unlabeled_samples.clone("rulb", num_unlabeled_samples, 0, num_unlabeled_samples);

    Samples* train_positive_samples = random_positive_samples->subset("tpos", num_positive_samples, 0);
    Samples* train_unlabeled_samples = random_unlabeled_samples->subset("tulb", num_unlabeled_samples, 0);

    Samples* unlabeled_positive_samples = random_positive_samples->subset("upos", num_unlabeled_positive_samples, num_positive_samples);

    test(*train_positive_samples, *train_unlabeled_samples, *unlabeled_positive_samples, test_samples);

    //int num_unlabeled_unlabeled_samples = (double)NUM_UNLABELED_POSITIVE_SAMPLES * UNLABELED_UNLABELED_RATIO;
    //Samples* samples2 = unlabeled_samples.clone("smp2", num_unlabeled_samples + num_unlabeled_unlabeled_samples);
    //Samples* unlabeled_unlabeled_samples = samples2->subset("uulb", num_unlabeled_unlabeled_samples, num_unlabeled_samples);
    //test(*train_positive_samples, *train_unlabeled_samples, *unlabeled_positive_samples, *unlabeled_unlabeled_samples);

    delete train_positive_samples;
    delete train_unlabeled_samples;
    delete unlabeled_positive_samples;
    //delete unlabeled_unlabeled_samples;
    delete random_positive_samples;
    delete random_unlabeled_samples;
}

// std::tuple<bias, w>
typedef std::tuple<float64_t, SGVector<float64_t> > LibLinearModelParameters;
LibLinearModelParameters load_model(const std::string& model_filename)
{
    LibLinearModelParameters model;

    std::ifstream model_file(model_filename);

    float64_t bias;
    std::string lineBias;
    if (std::getline(model_file, lineBias) ){
        bias = atof(lineBias.c_str());

        std::vector<float64_t> weights;
        std::string lineW;
        if (std::getline(model_file, lineW) ){
            std::stringstream ss(lineW);
            
            std::string strW;
            while ( std::getline(ss, strW, ' ') ){
                float64_t a = atof(strW.c_str());
                weights.push_back(a);
            }
        }

        SGVector<float64_t> w(weights.size());
        for ( int i = 0 ; i < weights.size() ; i++ ){
            w[i] = weights[i];
        }

        model = std::make_tuple(bias, w);
    }

    model_file.close();

    return model;
}

void predict_samples(Samples& samples)
{
    samples.load();

    CLibLinear* liblinear_machine= new CLibLinear();

    float64_t bias;
    SGVector<float64_t> w;
    std::tie(bias, w) = load_model("./data/pu.model");

    liblinear_machine->set_bias(bias);
    liblinear_machine->set_w(w);

    SGSparseMatrix<float64_t>* test_data_mat = samples.get_tfidf_matrix();
    SGVector<float64_t>* test_labels_vec = samples.get_sg_labels();
    CSparseFeatures<float64_t>* test_data = new CSparseFeatures<float64_t>(*test_data_mat);
    CBinaryLabels* test_labels = new CBinaryLabels(*test_labels_vec);

    CBinaryLabels* pred_labels = liblinear_machine->apply_binary(test_data);
    CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();
    float64_t accuracy = predict_accuracy(eval, pred_labels, test_labels);
    info_log("Predict accuracy: %.3f", accuracy);

    std::vector<double> pred_result;
    for ( size_t i = 0 ; i < pred_labels->get_num_labels() ; i++ ){
        float64_t v = pred_labels->get_value(i);
        pred_result.push_back(v);
    }
    samples.print_pred_result(pred_result);


    SG_UNREF(liblinear_machine);
}

void query_sample(Samples& samples, int sample_id)
{
    samples.load_vocabulary();

    samples.query_sample(sample_id);
}

int testdb(const std::string& dbwords_dir, const std::string& strKey)
{
    leveldb::DB* dbwords;
    //std::string dbwords_dir = get_dbwords_dir();
    leveldb::Status status = leveldb::DB::Open(leveldb::Options(), dbwords_dir + "/words", &dbwords);
    if ( !status.ok() ){
        error_log("Open leveldb %s failed.", dbwords_dir.c_str());
        error_log("%s", status.ToString().c_str());
        return -1;
    }

    leveldb::Iterator* iter = dbwords->NewIterator(leveldb::ReadOptions());
    iter->SeekToFirst();
    while ( iter->Valid() ){
        leveldb::Slice slice_key = iter->key();
        leveldb::Slice slice_value = iter->value();

        std::string key = slice_key.ToString();
        std::string value = slice_value.ToString();

        std::cout << key << " ";

        iter->Next();
    };
    delete iter;
    std::cout << std::endl;

    //leveldb::DB* dbwords;
    //leveldb::Options options;
    ////options.create_if_missing = true;
    //leveldb::Status status = leveldb::DB::Open(options, dbwords_dir, &dbwords);
    //if ( !status.ok() ){
        //std::cout << status.ToString().c_str() << std::endl;
        //return -1;
    //}

    //leveldb::Iterator* iter = dbwords->NewIterator(leveldb::ReadOptions());
    //iter->SeekToFirst();
    //while ( iter->Valid() ){
        //leveldb::Slice slice_key = iter->key();
        //leveldb::Slice slice_value = iter->value();

        //std::string key = slice_key.ToString();
        //std::string value = slice_value.ToString();

        //std::cout << key << " ";

        //iter->Next();
    //};
    //std::cout << std::endl;
 
    leveldb::Slice key(strKey);
    std::string value;
    status = dbwords->Get(leveldb::ReadOptions(), key, &value);
    if ( !status.ok() ){
        std::cout << status.ToString().c_str() << std::endl;
        delete dbwords;
        return -1; 
    }

    delete dbwords;

    return 0;
}

/* ==================== usage() ==================== */
static void usage(int status)
{
    if ( status )
        printf("Try `%s --help' for more information.\n", program_name);
    else {
        printf("Usage: %s [OPTION]\n", program_name);
        printf("udb\n\
                -i, --import_samples    Import samples from other format data.\n\
                -b, --rebuild           Rebuild samples(do words segmentation...).\n\
                -l, --learning          Learn a model.\n\
                -p, --predict           Predict test dataset using model.\n\
                -x, --xls_file          XLS file.\n\
                -f, --samples_dir       Samples root directory.\n\
                -z, --test              Test.\n\
                -v, --verbose           print debug messages\n\
                -h, --help              display this help and exit\n\
                \n");
    }
    exit(status);
}
static struct option const long_options[] = {
    {"xls_file", required_argument, NULL, 'x'},
    {"samples_dir", required_argument, NULL, 'f'},
    {"sample_id", required_argument, NULL, 'e'},
    {"positive_samples", required_argument, NULL, 'o'},
    {"unlabeled_samples", required_argument, NULL, 'u'},
    {"test_samples", required_argument, NULL, 't'},
    {"import_samples", no_argument, NULL, 'i'},
    {"rebuild", no_argument, NULL, 'b'},
    {"learning", no_argument, NULL, 'l'},
    {"predict", no_argument, NULL, 'p'},
    {"query", no_argument, NULL, 'q'},
    {"test", no_argument, NULL, 'z'},
	{"verbose", no_argument, NULL, 'v'},
	{"help", no_argument, NULL, 'h'},

	{NULL, 0, NULL, 0},
};
static const char *short_options = "x:f:e:o:u:t:iblpqtvh";

int main(int argc, char *argv[])
{

    //if ( argc < 3 ){
        //std::cout << "Usage: " << argv[0] << " <positive_samples_dir> <unlabeled_samples_dir>" << std::endl;
        //exit(-1);
    //}

    int log_level = LOG_INFO;
    std::string action;
    std::string xls_file;
    std::string samples_dir;
    int sample_id;
    std::string positive_samples_dir, unlabeled_samples_dir, test_samples_dir;
    bool is_test = 0;
    /* -------- Program Options -------- */
	int ch, longindex;
	while ((ch = getopt_long(argc, argv, short_options, long_options, &longindex)) >= 0) {
        switch (ch) {
            case 'x':
                xls_file = optarg;
                break;
            case 'f':
                samples_dir = optarg;
                break;
            case 'e':
                sample_id = atoi(optarg);
                break;
            case 'o':
                positive_samples_dir = optarg;
                break;
            case 'u':
                unlabeled_samples_dir = optarg;
                break;
            case 't':
                test_samples_dir = optarg;
                break;
            case 'i':
                action = "import_samples";
                break;
            case 'b':
                action = "rebuild";
                break;
            case 'l':
                action = "learning";
                break;
            case 'p':
                action = "predict";
                break;
            case 'q':
                action = "query";
                break;
            case 'z':
                is_test = 1;
                break;
            case 'v':
                log_level = LOG_DEBUG;
                break;
            case 'h':
                usage(0);
                break;
            default:
                usage(1);
                break;
        }
    }

	if (optind != argc){
		//po.docset_path = argv[optind];
    }

    init_logger(program_name, LOG_DEBUG);

    notice_log("%s start.", program_name);

    GET_TIME_MILLIS(msec0);

    //std::string positive_samples_dir = argv[1];
    //std::string unlabeled_samples_dir = argv[2];

    //std::string test_samples_dir;
    //if ( argc > 3 ){
        //test_samples_dir = argv[3];
    //}

    init_shogun_with_defaults();
    sg_io->set_loglevel(MSG_INFO);

    if ( action == "import_samples" ){

    } else if ( action == "rebuild" ){
        Samples samples("", samples_dir);
        samples.rebuild_words();
    } else if ( action == "learning" ){
        Samples positive_samples("pos", positive_samples_dir);
        Samples unlabeled_samples("neg", unlabeled_samples_dir);
        Samples test_samples("test", test_samples_dir);
        learning(positive_samples, unlabeled_samples, test_samples);
    } else if ( action == "predict" ){
        Samples test_samples("test", samples_dir);
        predict_samples(test_samples);
    } else if ( action == "query" ){
        Samples samples("", samples_dir);
        query_sample(samples, sample_id);
    }

    //testdb(test_samples_dir, argv[4]);
    //test_samples.load_words();

    //rebuild_words(positive_samples, unlabeled_samples, test_samples);

    //learning(positive_samples, unlabeled_samples, test_samples);

    //predict_samples(test_samples);

    //int sample_id = atoi(argv[4]);
    //query_sample(test_samples, sample_id);
    //query_sample(positive_samples, sample_id);

    exit_shogun();

    GET_TIME_MILLIS(msec_end);

    notice_log("Total Time: %zu.%03zu sec.", (size_t)(msec_end - msec0) / 1000, (size_t)(msec_end - msec0) % 1000);

    return 0;
}

