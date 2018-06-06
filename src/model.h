#pragma once

#include <memory>
#include <mutex>

#include "args.h"
#include "vector.h"
#include "real.h"

namespace poincare {

static const real BALL_MAX_DISTANCE = 1 - 1e-5; // for the Nickel & Kiela style updates

class Model {
    protected:
        std::shared_ptr<std::vector<Vector>> vectors_;
        std::shared_ptr<Args> args_;
        real performance_;
        int64_t nexamples_;

    public:
        int64_t update_count;
        int64_t pullback_count;

        Model(std::shared_ptr<std::vector<Vector>> vectors, std::shared_ptr<Args> args);

        void nickel_kiela_objective(int32_t source, std::vector<int32_t>& samples, real lr);

        /**
         * Return a metric on the average performance of this model since the last
         * call to this function (so this function is not idempotent).
         */
        real get_performance();

        /**
         * Update (in place) the hyperboloid point in the direction of its
         * (hyperboloid-)tangent vector `tangent`.    If args_->additive_updates,
         * then update by projecting both the point and the tangent to the Poincare
         * ball, then adding the tangent vector `tangent` and pulling back inside
         * the ball, if necessary (in the manner of Nickel & Kiela), before
         * returning to the hyperboloid.    Otherwise, use the exponential map on the
         * hyperboloid.
         */
        void update(Vector& point, Vector& tangent);
};

}
