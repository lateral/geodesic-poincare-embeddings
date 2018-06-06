#include "model.h"

namespace poincare {

constexpr real MAX_MINKOWSKI_DOT = -1 - 1e-10;
constexpr real MIN_STEP_SIZE = 1e-10;

Model::Model(std::shared_ptr<std::vector<Vector>> vectors, std::shared_ptr<Args> args) {
    vectors_ = vectors;
    args_ = args;
    performance_ = 0.0;
    nexamples_ = 1;
    update_count = 1;
    pullback_count = 0;
}

void Model::update(Vector& point, Vector& tangent) {
    update_count++;
    real tangent_norm = std::sqrt(std::max(minkowski_dot(tangent, tangent), (real)0));
    if (tangent_norm < MIN_STEP_SIZE) {
        return;
    }
    real step_size = tangent_norm;
    // normalize the tangent vector
    tangent.multiply(1.0 / tangent_norm);
    // clip the step size, if needed
    if (step_size > args_->max_step_size) {
        step_size = args_->max_step_size;
    }
    if (args_->additive_updates) {
        tangent.multiply(step_size);
        tangent.to_ball_tangent(point);
        point.to_ball_point();
        point.add(tangent);
        // pull back inside the ball if necessary
        real norm = std::sqrt(minkowski_dot(point, point));
        if (norm >= 1) {
            pullback_count++;
            point.multiply(BALL_MAX_DISTANCE / norm);
        }
        point.to_hyperboloid_point();
        point.ensure_on_hyperboloid();
    } else {
        // geodesic updates
        point.geodesic_update(tangent, step_size);
    }
}

void Model::nickel_kiela_objective(int32_t source, std::vector<int32_t>& samples, real lr) {
    Vector acc_source_gradient(args_->dimension + 1);
    Vector sample_gradient(args_->dimension + 1);
    // compute the minkowski dot product and activation for each sample
    // ... and also the normalisation factor, z.
    std::vector<real> mdps(samples.size());
    std::vector<real> activations(samples.size());
    real mdp;
    real activation;
    real z = 0;

    for (int32_t n = 0; n < samples.size(); n++) {
        mdp = minkowski_dot(vectors_->at(source), vectors_->at(samples[n]));
        if (mdp > MAX_MINKOWSKI_DOT) {
            mdp = MAX_MINKOWSKI_DOT;
        }
        mdps[n] = mdp;
        activation = 1. / (-1 * mdp + std::sqrt(pow(mdp, 2) - 1));
        activations[n] = activation;
        z += activation;
    }
    performance_ += activations[0] / z;

    for (int32_t n = 0; n < samples.size(); n++) {
        real label = (n == 0);
        real weight = (-label + activations[n] / z) * (-1. / std::sqrt(pow(mdps[n], 2) - 1));
        // accumulate the unprojected gradient for the input word vector
        acc_source_gradient.add(vectors_->at(samples[n]), weight);
        // update the output word vector
        sample_gradient = vectors_->at(source);
        sample_gradient.multiply(lr * weight);
        sample_gradient.project_onto_tangent_space(vectors_->at(samples[n]));
        update(vectors_->at(samples[n]), sample_gradient);
    }
    nexamples_ += 1;

    acc_source_gradient.multiply(lr);
    acc_source_gradient.project_onto_tangent_space(vectors_->at(source));
    update(vectors_->at(source), acc_source_gradient);
}

real Model::get_performance() {
    real avg = performance_ / nexamples_;
    performance_ = 0.0;
    nexamples_ = 1;
    return avg;
}

}
