#pragma once

#include <random>
#include <fstream>
#include <memory>

#include "args.h"
#include "digraph.h"
#include "sampler.h"
#include "model.h"
#include "real.h"
#include "vector.h"

namespace poincare {

static const int32_t NEGATIVE_TABLE_SIZE = 100000000; // increased from the original

class Poincare {
 protected:
    std::shared_ptr<Args> args_;
    std::shared_ptr<Digraph> digraph;
    std::shared_ptr<Sampler> sampler;

    std::shared_ptr<std::vector<Vector>> vectors_;
    std::shared_ptr<std::vector<std::mutex>> vector_flags_;

    std::shared_ptr<Model> model_;
    real performance;

    void save_checkpoint(int32_t epochs_trained, real performance);

    /**
     * Lock both the source and target; if this fails, return false; if it
     * succeeds, then proceed to lock the specified number of negative samples,
     * which are guaranteed to be distinct, and return true, in which case the
     * vector `samples` is populated with target, and then the negative samples.
     * If false is returned, then `samples` is unchanged.
     */
    bool obtain_vectors(int32_t source, int32_t target, std::vector<int32_t>& samples, std::minstd_rand& rng);

    /**
     * Release the locks of source and all the samples provided.
     */
    void release_vectors(int32_t source, std::vector<int32_t>& samples);

 public:
    Poincare(std::shared_ptr<Args> args);

    /**
     * Save the vectors to the filename specified (as points on
     * the Poincaré ball).
     */
    void save_vectors(std::string);

    /**
     * Given the filename of a CSV containing the vectors for
     * each node (as points on the Poincaré ball), load these
     * as model parameters, converting them to points on the
     * hyperboloid.
     */
    void load_vectors(std::string);
    void print_info(real, real);

    void epoch_thread(int32_t thread_id, uint32_t seed, real start_lr, real end_lr);
    void train();

};
}
