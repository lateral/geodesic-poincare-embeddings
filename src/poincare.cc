#include "poincare.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>

// how many tokens to process before reporting on performance
constexpr int32_t REPORTING_INTERVAL = 250;

namespace poincare {

    Poincare::Poincare(std::shared_ptr<Args> args) {
        args_ = args;
        performance = 0;
    }

void Poincare::save_vectors(std::string fn) {
    std::ofstream ofs(fn);
    if (!ofs.is_open()) {
        throw std::invalid_argument(fn + " cannot be opened!");
    }
    for (int32_t i = 0; i < digraph->node_count(); i++) {
        std::string name = (digraph->enumeration2node[i])->name;
        Vector vector(vectors_->at(i)); // a copy
        vector.to_ball_point();
        ofs << name << " " << vector << std::endl;
    }
    ofs.close();
}

void Poincare::load_vectors(std::string fn) {
    std::ifstream in(fn);
    if (!in.is_open()) {
        throw std::invalid_argument(fn + " cannot be opened!");
    }
    std::string line;
    while (std::getline(in, line)) {
        std::string field;
        std::stringstream line_stream(line);
        // count the fields
        int col = 0;
        int32_t offset;
        while (std::getline(line_stream, field, ' ')) {
            if (col == 0) {
                offset = (digraph->name2node).at(field)->enumeration;
            } else {
                vectors_->at(offset)[col - 1] = std::stold(field);
            }
            col++;
        }
        vectors_->at(offset).to_hyperboloid_point();
    }
    in.close();
}

void Poincare::print_info(real progress, real lr) {
    if (args_->verbose) {
        std::cerr << std::fixed;
        std::cerr << "\r" << std::setw(5) << std::setprecision(1) << 100 * progress << "%";
        std::cerr << std::setfill(' ');
        std::cerr << "  lr: " << std::setw(5) << std::setprecision(3) << lr;
        std::cerr << std::flush;
    }
}

bool Poincare::obtain_vectors(int32_t source, int32_t target, std::vector<int32_t>& samples, std::minstd_rand& rng) {
    if (!vector_flags_->at(source).try_lock()) {
        return false;
    }
    if (!vector_flags_->at(target).try_lock()) {
        vector_flags_->at(source).unlock();
        return false;
    }
    samples.clear();
    samples.push_back(target);

    while (samples.size() < args_->number_negatives + 1) {
        Node* source_node = (digraph->enumeration2node)[source];
        auto next_negative = sampler->get_sample(source_node->target_enums, rng);
        if (vector_flags_->at(next_negative).try_lock()) {
            samples.push_back(next_negative);
        }
    }
    return true;
}

void Poincare::release_vectors(int32_t source, std::vector<int32_t>& samples) {
    for (int32_t n = 0; n < samples.size(); n++) {
        vector_flags_->at(samples[n]).unlock();
    }
    vector_flags_->at(source).unlock();
}

void Poincare::epoch_thread(int32_t thread_id, uint32_t seed, real start_lr, real end_lr) {
    std::minstd_rand rng(1 + seed); // seed 0 and 1 coincide for minstd_rand
    const int64_t edges_per_thread = digraph->edges.size() / args_->threads;
    Model model(vectors_, args_);

    int64_t iter_count = 0; // number processed so far
    int64_t skipped = 0; // number skipped due to locking
    real lr = start_lr;
    real progress = 0.;
    std::vector<int32_t> samples;
    Edge* edge;
    for (int64_t i = thread_id; i < digraph->edges.size(); i=i+args_->threads) {
        edge = (digraph->edges)[i];
        iter_count++;
        int32_t source_enum = (edge->source).enumeration;
        int32_t target_enum = (edge->target).enumeration;
        progress = real(iter_count) / edges_per_thread;
        lr = start_lr * (1.0 - progress) + end_lr * progress;
        samples.clear();
        if (!obtain_vectors(source_enum, target_enum, samples, rng)) {
            // couldn't obtain one of the necessary locks, so skip!
            skipped++;
            continue;
        }
        model.nickel_kiela_objective(source_enum, samples, lr);
        release_vectors(source_enum, samples);
        if (thread_id == 0) {
            // only thread 0 is responsible for printing progress info
            if (iter_count % REPORTING_INTERVAL == 0) {
                print_info(progress, lr);
            }
        }
    }
    performance += model.get_performance();
    if (thread_id == 0) {
        print_info(progress, lr);
        std::cerr << std::endl;
        std::cerr << std::setfill('0');
        std::cerr << "Thread 0: skipped " << std::setw(6) << skipped << "/" << std::setw(6) << iter_count << " problems; ";
        std::cerr << "pullbacks for " << std::setw(6) << model.pullback_count << "/" << std::setw(6) << model.update_count << " updates.\n";
    }
}

void Poincare::train() {
    std::ifstream ifs(args_->graph);
    if (!ifs.is_open()) {
        throw std::invalid_argument(args_->graph + " cannot be opened!");
    }
    digraph = std::make_shared<Digraph>(ifs);
    ifs.close();
    
    // setup the negative sampler
    std::vector<int64_t> counts(digraph->node_count());
    for (int i=0; i < digraph->node_count(); i++) {
        counts[i] = (digraph->enumeration2node)[i]->count_as_target;
    }
    std::cerr << "Generating negative samples...\n";
    sampler = std::make_shared<Sampler>(args_->distribution_power, counts, NEGATIVE_TABLE_SIZE);
    // initialise the vectors
    std::minstd_rand rng(args_->seed);
    Vector init_vector(args_->dimension + 1);
    vectors_ = std::make_shared<std::vector<Vector>>();
    for (int64_t i=0; i < digraph->node_count(); i++) {
        random_hyperboloid_point(init_vector, rng, args_->init_std_dev);
        vectors_->push_back(init_vector);
    }
    // overwrite the init vectors with any pre-trained vectors
    if (!(args_->input_vectors).empty()) {
        std::cerr << "Loading vectors: " << args_->input_vectors << "\n";
        load_vectors(args_->input_vectors);
    }
    vector_flags_ = std::shared_ptr<std::vector<std::mutex>>(new std::vector<std::mutex>(vectors_->size()));
    // start the training!
    real lr_delta_per_epoch = (args_->start_lr - args_->end_lr) / args_->epochs;;
    for (int32_t epoch = 0; epoch < args_->epochs; epoch++) {
        save_checkpoint(epoch, performance);
        std::cerr << "\n" << std::string(80, '-') << "\n\n";
        std::cerr << "\rEpoch: " << (epoch + 1) << " / " << args_->epochs;
        std::cerr << std::flush;
        real epoch_start_lr = args_->start_lr - real(epoch) * lr_delta_per_epoch;
        real epoch_end_lr = args_->start_lr - real(epoch + 1) * lr_delta_per_epoch;
        std::vector<std::thread> threads;
        performance = 0;
        clock_t start = clock();
        for (int32_t thread_id = 0; thread_id < args_->threads; thread_id++) {
            int32_t thread_seed = args_->seed + epoch * args_->threads + thread_id;
            threads.push_back(std::thread([=]() {
                epoch_thread(thread_id, thread_seed, epoch_start_lr, epoch_end_lr);
            }));
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
        performance /= args_->threads;
        real cpu_time_single_thread = real(clock() - start) / (CLOCKS_PER_SEC * args_->threads);
        std::cerr << std::setfill(' ');
        std::cerr << "Epoch took " << std::setw(5) << std::setprecision(3) << cpu_time_single_thread << " seconds; ";
        std::cerr << "mean objective " << std::setw(5) << std::setprecision(3) << performance << "\n";
        std::cerr << std::flush;
    }
    save_checkpoint(args_->epochs, performance);
}

void Poincare::save_checkpoint(int32_t epochs_trained, real performance) {
    if (args_->checkpoint_interval > 0 && epochs_trained % args_->checkpoint_interval == 0) {
        // checkpoint (save) the vectors - pad epoch number to maintain
        // alphabetical ordering
        std::ostringstream out;
        out << args_->output_vectors;
        out << "-after-" << std::setfill('0') << std::setw(6) << epochs_trained << "-epochs";
        out << "-objective-" << std::setprecision(3) << performance;
        std::string fn = out.str();
        std::cerr << "Saving checkpoint " << fn << "\n";
        this->save_vectors(fn);
    }
}

}
