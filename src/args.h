#pragma once

#include <string>
#include <vector>

namespace poincare {

class Args {
    public:
        Args();
        std::string graph;
        std::string input_vectors;
        std::string output_vectors;
        double max_step_size;
        double start_lr;
        double end_lr;
        int seed;
        int dimension;
        int checkpoint_interval;
        double distribution_power;
        int epochs;
        int number_negatives;
        int threads;
        double init_std_dev;
        bool additive_updates;
        bool verbose;

    void parse_args(const std::vector<std::string>& args);
    void print_help();
};
}
