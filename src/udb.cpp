#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <msgpack.hpp>
#include <tuple>
#include <shogun/base/init.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/machine/BaggingMachine.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
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


namespace udb{
    typedef SGSparseMatrix<float64_t> DataMatrix;
    typedef SGVector<float64_t> Labels;
    typedef SGMatrix<float64_t> RealMatrix;
    typedef SGVector<float64_t> RealVector;

}; // namespace udb

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
public:
    Vocabulary(){
    }

    ~Vocabulary(){
    }

    // -------------------- clear() --------------------
    void clear(){
        m_mapIdWords.clear();
        m_mapWordIds.clear();
    }

    // -------------------- add_word() --------------------
    int add_word(const std::string& word, const std::string& pos){
        int id = -1;
        std::map<std::string, int>::const_iterator it;
        it = m_mapWordIds.find(word);
        if ( it != m_mapWordIds.end() ){
            id = it->second;
        } else {
            id = size();
            m_mapIdWords[id] = make_word(word, pos);
            m_mapWordIds[word] = id;
        }

        return id;
    }

    // -------------------- size() --------------------
    size_t size() const {
        return m_mapIdWords.size();
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

const char *jieba_dict = "../jieba/dict/jieba.dict.utf8";
const char *hmm_model = "../jieba/dict/hmm_model.utf8";
const char *user_dict = "../jieba/dict/user.dict.utf8";

//cppjieba::Jieba *jieba = NULL;
cppjieba::PosTagger *jieba = NULL;

// ==================== class Samples ===================
class Samples{
public:
    typedef std::vector<double> Labels;

    // std::list<int term_id>
    typedef std::list<int> WordsList;
    // std::tuple<int sample_id, WordsList>
    typedef std::tuple<int, WordsList> SampleWords;
    typedef std::list<SampleWords> SampleWordsList;

    // -------------------- make_sample_words() --------------------
    static SampleWords make_sample_words(int sample_id, const WordsList& words){
        return std::make_tuple(sample_id, words);
    }

    // -------------------- print_sample_words() --------------------
    void print_sample_words(const SampleWords& sample_words) const
    {
        int sample_id;
        WordsList words;
        std::tie(sample_id, words) = sample_words;

        std::cout << "[" << sample_id << "] ";
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
            WordsList words;
            std::tie(sample_id, words) = *it;

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

        std::cout << "SampleWordsList -> SampleTermsList (" << sample_terms_list.size() << " samples)." << std::endl;

        return 0;
    }

    udb::DataMatrix* sample_terms_to_data_matrix(const SampleTermsList& sample_terms_list) const{
        size_t num_feat = m_vocabulary.size();
        size_t num_vec = sample_terms_list.size();

        udb::DataMatrix* mat = new udb::DataMatrix(num_feat, num_vec);

        int idx = 0;
        for ( SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++, idx++ ){
            TermFrequencies term_frequencies;
            std::tie(std::ignore, std::ignore, term_frequencies) = *it;
            for ( TermFrequencies::const_iterator it0 = term_frequencies.begin() ; it0 != term_frequencies.end() ; it0++ ){
                int term_id = it0->first;
                int term_used = it0->second;
                (*mat)(idx, term_id) = term_used;
            }
            
        }
        return mat;
    }

private:
    std::string m_samplesdir;
    Vocabulary m_vocabulary;
    SampleWordsList m_sample_words_list;
    Labels m_labels;

public:
    Samples(){
    }

    Samples(const std::string& samples_dir)
    : m_samplesdir(samples_dir){
    }

    ~Samples(){
    }

    size_t get_num_samples() const {
        return m_sample_words_list.size();
    }

    size_t get_num_features() const{
        return m_vocabulary.size();
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
        for ( int i = 0 ; i < get_num_samples() ; i++ ){
            m_labels.push_back(value);
        }
    }

    SGVector<float64_t> get_sg_labels(){
        size_t num_features = m_labels.size();
        SGVector<float64_t> sglabels(num_features);
        for ( int i = 0 ; i < num_features ; i++ ){
            sglabels.set_element(i, m_labels[i]);
        }
        return sglabels;
    }

    SGMatrix<float64_t> get_sg_matrix(){
        std::cout << "get_sg_matrix()..." << std::endl;

        SampleTermsList sample_terms_list;
        sample_words_to_sample_terms(m_sample_words_list, sample_terms_list);
        //print_sample_terms_list(sample_terms_list);

        size_t num_features = get_num_features();
        size_t num_vectors = sample_terms_list.size();

        SGMatrix<float64_t> mat(num_features, num_vectors);

        int idx = 0;
        for ( Samples::SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++, idx++ ){
            Samples::TermFrequencies term_frequencies;
            std::tie(std::ignore, std::ignore, term_frequencies) = *it;

            for ( Samples::TermFrequencies::const_iterator it0 = term_frequencies.begin() ; it0 != term_frequencies.end() ; it0++ ){
                int term_id = it0->first;
                int term_used = it0->second;
                mat(term_id, idx) = term_used;
            }
        }
        std::cout << "num_rows: " << mat.num_rows << " num_cols: " << mat.num_cols << std::endl;

        return mat;
    }


    int find_samples_has_term(const Samples::SampleTermsList& sample_terms_list, int term_id){
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

    // TODO
    SGMatrix<float64_t>* get_tfidf_matrix(bool use_idf = true){
        int num_samples = get_num_samples();
        int num_features = get_num_features();

        SGMatrix<float64_t>* tfidf_mat = new SGMatrix<float64_t>(num_features, num_samples);

        int row = 0;
        std::map<int, int> mapTermSamples;
        for ( SampleTermsList::const_iterator it = sample_terms_list.begin() ; it != sample_terms_list.end() ; it++ , row++){
            int total_terms_used;
            TermFrequencies term_frequencies;
            std::tie(std::ignore, total_terms_used, term_frequencies) = *it;

            for ( TermFrequencies::const_iterator it0 = term_frequencies.begin() ; it0 != term_frequencies.end() ; it0++ ){
                double value = 0.0;

                int term_id = it0->first;
                int term_used = it0->second;
                double tf = term_used / (total_terms_used);
                double idf = 1.0;
                if ( use_idf ){
                    int samples_has_term = 0;
                    std::map<int, int>::iterator it1 = mapTermSamples.find(term_id);
                    if ( it1 != mapTermSamples.end() ){
                        samples_has_term = it1->second;
                    } else {
                        samples_has_term = find_samples_has_term(sample_terms_list, term_id); 
                        mapTermSamples[term_id] = samples_has_term;
                    }

                    idf = log(num_samples / (samples_has_term + 1));
                }

                tfidf_mat(term_id, row) = tf * idf;
            }
        }

        return tfidf_mat;
    }

    int load(){
        std::cout << "### Load " << m_samplesdir << std::endl;

        std::cout << "Loading vocabulary... ";
        load_vocabulary();
        std::cout << " (" << get_num_features() << " features loaded.)" << std::endl;

        std::cout << "Loading words...";
        load_words();
        //print_sample_words_list(sample_words_list);
        size_t num_samples = get_num_samples();
        std::cout << " (" << num_samples << " samples loaded.)" << std::endl;

        // labels
        init_labels(1.0);

        std::cout << "Done." << std::endl << std::endl;

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

    // -------------------- rebuild_words() --------------------
    // create samples_dir/words from samples_dir/content.
    int rebuild_words(){

        if ( jieba == NULL ){
            jieba = new cppjieba::PosTagger(jieba_dict, hmm_model, user_dict);
        }

        leveldb::Options options;
        leveldb::Status status;
        
        leveldb::DB* dbcontent;
        std::string dbcontent_dir = get_dbcontent_dir();
        status = leveldb::DB::Open(options, dbcontent_dir, &dbcontent);
        if ( !status.ok() ) {
            std::cout << "Open leveldb " << dbcontent_dir << " failed." << std::endl;
            return -1;
        }

        leveldb::DB* dbwords;
        options.create_if_missing = true;
        std::string dbwords_dir = get_dbwords_dir();
        status = leveldb::DB::Open(options, dbwords_dir, &dbwords);
        if ( !status.ok() ){
            std::cout << "Open leveldb " << dbwords_dir << " failed." << std::endl;
            return -1;
        }
        leveldb::WriteBatch batchWords;

        std::string words_filename = get_words_filename();
        std::ofstream words_file(words_filename, std::ios::out | std::ios::trunc);

        m_vocabulary.clear();

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
                std::cout << "sample " << key << " msgpack type mismatched." << std::endl;
            }

            int sample_id;
            std::string title, content;
            msgpack::type::tie(sample_id, title, content) = dst;

            std::cout << "=============================" << std::endl;
            std::cout << "sample_id: " << sample_id << std::endl;
            std::cout << "title: " << title << std::endl;
           
            std::stringstream ss;
            ss  << sample_id << " ";

            WordsList words = do_word_segmentation(title + "\n" + content);

            // TODO
            std::string str_sample_words = serialize_sample_words(sample_id, words);
            std::stringstream ss1;
            ss1 << sample_id;
            batchWords.Put(ss.str(), str_sample_words);

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
        m_sample_words_list.clear();

        leveldb::DB* dbwords;
        std::string dbwords_dir = get_dbwords_dir();
        leveldb::Status status = leveldb::DB::Open(leveldb::Options(), dbwords_dir, &dbwords);
        if ( !status.ok() ){
            std::cout << "Open leveldb " << dbwords_dir << " failed." << std::endl;
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
            m_sample_words_list.push_back(sample_words);

            iter->Next();
        };
        delete iter;

        delete dbwords;

        return 0;
    }

    // -------------------- load_vocabulary() --------------------
    int load_vocabulary(){
        m_vocabulary.clear();
        std::string vocabulary_filename = get_vocabulary_filename();
        std::ifstream vocabulary_file(vocabulary_filename);

        std::string line;
        while ( std::getline(vocabulary_file, line) ){
            std::stringstream ss(line);

            std::string str_term_id;
            std::getline(ss, str_term_id, ' ');
            int term_id = atoi(str_term_id.c_str());

            std::string word;
            std::getline(ss, word, ' ');

            std::string pos;
            std::getline(ss, pos, ' ');

            m_vocabulary.add_word(word, pos);
            //std::cout << term_id << "," << word << "," << pos << std::endl;
        };

        return 0;
    }


    int merge(const Samples& samples){
        std::map<int, int> idmap = m_vocabulary.merge(samples.m_vocabulary);

        int sample_id = get_num_samples();
        SampleWordsList::const_iterator it;
        for ( it = samples.m_sample_words_list.begin() ; it != samples.m_sample_words_list.end() ; it++ ){
            WordsList words;
            std::tie(std::ignore, words) = *it;

            for ( WordsList::iterator it0 = words.begin() ; it0 != words.end() ; it0++ ){
                int word = *it0;
                std::map<int, int>::const_iterator it_idmap = idmap.find(word);
                if ( it_idmap != idmap.end() ){
                    int word_new = it_idmap->second;
                    *it0 = word_new;
                } else {
                    std::cout << "Error: word id " << word << " does not exist." << std::endl; 
                    exit(-1);
                }
            }

            m_sample_words_list.emplace_back(sample_id++, words);
        }

        return 0;
    }

private:

    // -------------------- do_word_segmentation() --------------------
    WordsList do_word_segmentation(const std::string& content){
        WordsList words;

        typedef std::vector<std::pair<std::string, std::string> > Tags;
        Tags tags;
        jieba->Tag(content.c_str(), tags);
        
        for ( Tags::iterator it = tags.begin() ; it != tags.end() ; it++ ){
            std::pair<std::string, std::string>& tag = *it;
            std::string word = tag.first;
            std::string pos = tag.second;
            if ( word.length() <= 3 ) continue;
            //std::cout << "word " << word << " length() = " << word.length() << std::endl; 
            if ( (pos.substr(0, 1) == "n" && pos != "ns") || pos == "v" || pos == "vn" ) {
                int term_id = m_vocabulary.add_word(word, pos);    
                words.push_back(term_id);
            }
        }

        return words;
    }

    // -------------------- serialize_sample_words() --------------------
    std::string serialize_sample_words(int sample_id, const WordsList& words) const{
        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> packer(&sbuf);

        packer.pack(sample_id);
        int num_words = words.size();
        packer.pack_array(num_words);
        for ( WordsList::const_iterator it = words.begin() ; it != words.end() ; it++ ){
            int term_id = *it;
            packer.pack(term_id);
        }

        return std::string(sbuf.data(), sbuf.size());
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

        msgpack::unpacked result;
        if (!upk.next(&result)){
            std::cout << "No sample_id in msgpack buffer." << std::endl;
            return -1;
        }
        msgpack::object objSampleId = result.get();
        if ( objSampleId.type != msgpack::type::POSITIVE_INTEGER ){
            std::cout << "Wrong format for sample_id." << std::endl;
            return -1;
        }
        int sample_id = objSampleId.as<int>();
        //std::cout << "sample_id: " << sample_id << std::endl;

        if (!upk.next(&result)){
            std::cout << "No word list in msgpack buffer." << std::endl;
            return -1;
        }
        msgpack::object objWords = result.get();
        if ( objWords.type != msgpack::type::ARRAY ){
            std::cout << "Wrong format for word list." << std::endl;
            return -1;
        }
        int num_items = objWords.via.array.size;
        if ( objWords.via.array.size > 0 ){
            msgpack::object* pi = objWords.via.array.ptr;
            msgpack::object* pi_end = objWords.via.array.ptr + objWords.via.map.size;
            do {
                int term_id = pi[0].as<int>();
                words.push_back(term_id);
                //std::cout << term_id << " ";
                ++pi;
            } while (pi < pi_end);
        }
        //std::cout << std::endl;

        sample_words = make_sample_words(sample_id, words);
        return sample_words;
    }

}; // class Samples


// ==================== class Transformer ===================
class Transformer{
public:
    Transformer(){
    }

    virtual ~Transformer(){
    }

    udb::DataMatrix* fit_transform(udb::DataMatrix* X, udb::Labels* y = NULL){
        udb::DataMatrix* newMatrix = fit(X, y);
        if ( newMatrix != NULL ){
            return transform(newMatrix);
        } else {
            return NULL;
        }
    }

    virtual udb::DataMatrix* fit(udb::DataMatrix* X, udb::Labels* y = NULL){
        return X;
    }

    virtual udb::DataMatrix* transform(udb::DataMatrix* X, bool copy = true){
        udb::DataMatrix* X_copy = X;
        if ( copy ){
            X_copy = new udb::DataMatrix();
            *X_copy = *X;
        }

        return X_copy;
    }


}; // class Transformer


// ==================== class TfidfTransformer ===================
class TfidfTransformer : public Transformer{
private:
    std::string m_norm;
    bool m_use_idf;
    bool m_smooth_idf;
    bool m_subliner_tf;
public:
    TfidfTransformer(const std::string& norm="l2", bool use_idf=true, bool smooth_idf=true, bool subliner_tf=false) 
        : Transformer(), m_norm(norm), m_use_idf(use_idf), m_smooth_idf(smooth_idf), m_subliner_tf(subliner_tf){
    }

    virtual udb::DataMatrix* fit(udb::DataMatrix* X, udb::Labels* y = NULL){
        return X;
    }

    virtual udb::DataMatrix* transform(udb::DataMatrix* X, bool copy = true);

}; // class TfidfTransformer

udb::DataMatrix* TfidfTransformer::transform(udb::DataMatrix* X, bool copy){
    udb::DataMatrix* X_new = Transformer::transform(X, copy);

    for ( int i = 0 ; i < X_new->num_vectors ; i++ ){
        int num_terms = 0;
        for ( int j = 0 ; j < X_new->num_features ; j++ ){
            num_terms += (*X_new)(i,j);
        }
        for ( int j = 0 ; j < X_new->num_features ; j++ ){
            float a = (*X_new)(i,j);
            (*X_new)(i,j) = a / (num_terms);
        }
    }

    if ( m_use_idf ){
    }

    return X_new;
}


typedef std::tuple<float64_t, CEvaluation*, CLabels*> TrainResult;

TrainResult train_and_eval(CMachine* machine, CDenseFeatures<float64_t>* train_data, CBinaryLabels* train_labels)
{
    machine->train();

    CBinaryLabels* pred_labels = machine->apply_binary(train_data);
    pred_labels->get_int_labels().display_vector();

    CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();
    float64_t accuracy = eval->evaluate(pred_labels, train_labels);

    return std::make_tuple(accuracy, static_cast<CEvaluation*>(eval), static_cast<CLabels*>(pred_labels));
}

CMachine* build_liblinear_machine(CDenseFeatures<float64_t>* train_data, CBinaryLabels* train_labels)
{
    CLibLinear* liblinear_machine= new CLibLinear(2.0, train_data, train_labels);
    liblinear_machine->set_bias_enabled(true);

    float64_t accuracy;
    CEvaluation* eval;
    CBinaryLabels* pred_labels;
    TrainResult train_result = train_and_eval(liblinear_machine, train_data, train_labels);
    std::tie(accuracy, eval, (CLabels*&)pred_labels) = train_result;

    std::cout << "accuracy: " << accuracy << std::endl;

    SG_UNREF(eval);
    SG_UNREF(pred_labels);

    return liblinear_machine;
}

CMachine* build_bagging_machine(CDenseFeatures<float64_t>* train_data, CBinaryLabels* train_labels)
{
    int num_bags = 5;
    int bag_size = 25;
    CLibLinear* liblinear_machine = new CLibLinear();
    liblinear_machine->set_bias_enabled(true);
    CMajorityVote* mv = new CMajorityVote();

    CBaggingMachine* bagging_machine = new CBaggingMachine(train_data, train_labels);
    bagging_machine->set_num_bags(num_bags);
    bagging_machine->set_bag_size(bag_size);
    bagging_machine->set_machine(liblinear_machine);
    bagging_machine->set_combination_rule(mv);

    float64_t accuracy;
    CEvaluation* eval;
    CBinaryLabels* pred_labels;
    TrainResult train_result = train_and_eval(bagging_machine, train_data, train_labels);
    std::tie(accuracy, eval, (CLabels*&)pred_labels) = train_result;

    float64_t oob_error = bagging_machine->get_oob_error(eval);

    std::cout << "accuracy: " << accuracy << std::endl;
    std::cout << "oob error: : " << oob_error << std::endl;

    SG_UNREF(eval);
    SG_UNREF(pred_labels);

    return bagging_machine;
}

// TODO 
void BiasedSVM(const SGMatrix<float64_t>& data_mat, const SGVector<float64_t>& labels_vec){

    CDenseFeatures<float64_t>* train_data = new CDenseFeatures<float64_t>(data_mat);
    CBinaryLabels* train_labels = new CBinaryLabels(labels_vec);

    //CMachine* bagging_machine = build_bagging_machine(train_data, train_labels);
    //SG_UNREF(bagging_machine);

    CMachine* liblinear_machine = build_liblinear_machine(train_data, train_labels);
    liblinear_machine->print_serializable("ll_");
    SG_UNREF(liblinear_machine);

    SG_UNREF(train_data);
    SG_UNREF(train_labels);

}


void rebuild_words(Samples& positive_samples, Samples& unlabeled_samples){
    positive_samples.rebuild_words();
    unlabeled_samples.rebuild_words();
}

void test(Samples& positive_samples, Samples& unlabeled_samples)
{

    positive_samples.load();
    positive_samples.init_labels(1.0);
    size_t num_positive_samples = positive_samples.get_num_samples();
    unlabeled_samples.load();
    unlabeled_samples.init_labels(-1.0);
    size_t num_unlabeled_samples = unlabeled_samples.get_num_samples();

    Samples samples = positive_samples;
    std::cout << "Clone. samples: " << samples.get_num_samples() << " featuers: " << samples.get_num_features() << std::endl;

    samples.merge(unlabeled_samples);
    std::cout << "Merge. samples: " << samples.get_num_samples() << " featuers: " << samples.get_num_features() << std::endl;

    std::cout << "BiasedSVM..." << std::endl;
    SGMatrix<float64_t> data_mat = samples.get_sg_matrix();
    SGVector<float64_t> labels_vec = samples.get_sg_labels();
    //BiasedSVM(data_mat, labels_vec);

}

int main(int argc, char *argv[])
{
    if ( argc < 3 ){
        std::cout << "Usage: " << argv[0] << " <positive_samples_dir> <unlabled_samples_dir>" << std::endl;
        exit(-1);
    }

    std::string positive_samples_dir = argv[1];
    std::string unlabled_samples_dir = argv[2];

    init_shogun_with_defaults();
	//init_shogun(&udb_print_message, &udb_print_warning,
			//&udb_print_error, &udb_cancel_computations);
    sg_io->set_loglevel(MSG_DEBUG);

    //SG_SDEBUG("Start with %s\n", argv[1]);
    //SG_SWARNING("Start with %s\n", argv[1]);
    //SG_SERROR("Start with %s\n", argv[1]);

    Samples positive_samples(positive_samples_dir);
    Samples unlabeled_samples(unlabled_samples_dir);

    //rebuid_words(positive_samples, unlabeled_samples);
    test(positive_samples, unlabeled_samples);

    exit_shogun();

    return 0;
}

