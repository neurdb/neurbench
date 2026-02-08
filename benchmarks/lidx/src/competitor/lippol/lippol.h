#include"./src/src/core/lipp.h"
#include"../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class LIPPOLInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    void init(Param *param = nullptr) {}

    void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool remove(KEY_TYPE key, Param *param = nullptr);

    size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair <KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr);

    long long memory_consumption() { return lipp.total_size(); }

    PredictionStats get_prediction_stats() override {
        PredictionStats stats;
        stats.supported = true;
        // Get tree depth statistics - deeper tree means worse prediction
        auto [max_depth, avg_depth, num_leaves] = lipp.get_depth_stats();
        stats.avg_error = avg_depth;
        stats.max_error = max_depth;
        stats.total_lookups = num_leaves;
        return stats;
    }

private:
    lippolc::LIPP<KEY_TYPE, PAYLOAD_TYPE> lipp;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                      Param *param) {
    lipp.bulk_load(key_value, static_cast<int>(num));
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
    return lipp.at(key, val);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    lipp.insert(key, value);
    return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return lipp.update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
    return lipp.remove(key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair <KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
    return lipp.range_query_len(result, key_low_bound, static_cast<int>(key_num));
}