#include "gtest/gtest.h"
#include "vector.h"
#include "real.h"
#include <cmath>
#include <random>

namespace {

TEST(VectorTest, init_with_zeros) {
    int m = 5;
    poincare::Vector vec(m);
    vec.zero();
    EXPECT_EQ(vec.dimension_, m);
    for (auto i = 0; i < vec.dimension_; ++i) {
        EXPECT_EQ(0., vec[i]);
    }
}

TEST(VectorTest, multiply) {
    poincare::Vector vec(2);
    vec[0] = 1.;
    vec[1] = 2.;
    vec.multiply(1.5);
    EXPECT_FLOAT_EQ(1.5, vec[0]);
    EXPECT_FLOAT_EQ(3., vec[1]);
}

TEST(VectorTest, TestDot) {
    poincare::Vector vec0(2);
    vec0[0] = 1.;
    vec0[1] = 2.;
    poincare::Vector vec1(2);
    vec1[0] = 0.;
    vec1[1] = 4.;
    EXPECT_FLOAT_EQ(8., dot(vec0, vec1));
}

TEST(VectorTest, TestSquaredNorm) {
    poincare::Vector vec0(2);
    vec0[0] = 1.;
    vec0[1] = 2.;
    EXPECT_FLOAT_EQ(5., vec0.squared_norm());
}

TEST(VectorTest, minkowskiDot) {
    poincare::Vector vec_a(3);
    poincare::Vector vec_b(3);

    vec_a[0] = 1.;
    vec_a[1] = 0.5;
    vec_a[2] = -2.;

    vec_b[0] = 0.;
    vec_b[1] = 0.5;
    vec_b[2] = 1.;

    auto mdp = minkowski_dot(vec_a, vec_b);
    EXPECT_FLOAT_EQ(2.25, mdp);
}

TEST(VectorTest, randomHyperboloidPoint) {
    std::minstd_rand rng(1);
    poincare::Vector vec_a(3);
    poincare::Vector vec_b(3);

    random_hyperboloid_point(vec_a, rng, 0.1);
    random_hyperboloid_point(vec_b, rng, 0.1);
    EXPECT_NE(vec_a[0], vec_b[0]);    // vectors should be different

    // vectors should be on the hyperboloid
    auto mdp = minkowski_dot(vec_a, vec_a);
    EXPECT_FLOAT_EQ(-1., mdp);
}

TEST(VectorTest, distance) {
    poincare::Vector vec_a(2);
    poincare::Vector vec_b(2);

    // basepoint
    vec_a[0] = 0.;
    vec_a[1] = 1.0;

    real hyperangle = 0.5;
    vec_b[0] = std::sinh(hyperangle);
    vec_b[1] = std::cosh(hyperangle);

    real dist = distance(vec_a, vec_b);
    EXPECT_FLOAT_EQ(hyperangle, dist);
}

TEST(VectorTest, ensureOnHyperboloid) {
    poincare::Vector vec(2);
    
    // almost the basepoint
    vec[0] = 0.;
    vec[1] = 1.000001;

    // should now be the basepoint
    vec.ensure_on_hyperboloid();
    EXPECT_FLOAT_EQ(0.0, vec[0]);
    EXPECT_FLOAT_EQ(1.0, vec[1]);
}

TEST(VectorTest, ensureOnHyperboloidNoOp) {
    poincare::Vector vec(2);
    
    // basepoint: already on the hyperboloid
    vec[0] = 0.;
    vec[1] = 1.0;
    vec.ensure_on_hyperboloid();
    // nothing should have changed
    EXPECT_FLOAT_EQ(0., vec[0]);
    EXPECT_FLOAT_EQ(1., vec[1]);
}

TEST(VectorTest, toBallPointAtBasepoint) {
    poincare::Vector vec(2);
    // basepoint
    vec[0] = 0.;
    vec[1] = 1.0;
    vec.to_ball_point();
    // should be centre of Poincaré disc
    EXPECT_FLOAT_EQ(0., vec[0]);
    EXPECT_FLOAT_EQ(0., vec[1]);
}

TEST(VectorTest, toBallPoint) {
    poincare::Vector vec(2);
    real dist = 1;
    vec[0] = std::sinh(dist);
    vec[1] = std::cosh(dist);

    vec.to_ball_point();
    real norm = std::sqrt(minkowski_dot(vec, vec));
    EXPECT_FLOAT_EQ(std::tanh(dist / 2), norm);
}

TEST(VectorTest, toHyperboloidPoint) {
    poincare::Vector vec(3);
    real dist = 1.2;
    vec[0] = 0.;
    vec[1] = std::tanh(dist / 2);
    vec[2] = 0;
    vec.to_hyperboloid_point();
    EXPECT_FLOAT_EQ(0., vec[0]);
    EXPECT_FLOAT_EQ(std::sinh(dist), vec[1]);
    EXPECT_FLOAT_EQ(std::cosh(dist), vec[2]);
}

TEST(VectorTest, toBallTangent) {
    // a point on the hyperboloid
    poincare::Vector point(3);
    real dist = 1.2;
    point[0] = std::sinh(dist);
    point[1] = 0.;
    point[2] = std::cosh(dist);

    // a unit tangent vector in its tangent space
    poincare::Vector tangent(3);
    tangent[0] = 0.;
    tangent[1] = 1.;
    tangent[2] = 0.;
    
    // map to the corresponding tangent vector in the tangent space of the ball point
    tangent.to_ball_tangent(point);
    
    // check some obvious co-ordinates
    EXPECT_FLOAT_EQ(0., tangent[0]); // since hasn't changed rotational angle
    EXPECT_FLOAT_EQ(0., tangent[2]); // since it is tangent to the poincare disc

    // check its length
    real r = std::tanh(dist / 2); // displacement of corres. ball point from origin
    real euclid_norm = std::sqrt(minkowski_dot(tangent, tangent));
    // Euclidean norm is related to the tangent norm via r
    // Should be 1., since it is a unit vector in the tangent space of the
    // poincare disc (since it was a unit vector in the tangent space of the
    // hyperboloid, and the metric on the disc is induced).
    EXPECT_FLOAT_EQ(1., 2 * euclid_norm / (1 - r * r));
}

TEST(VectorTest, toHyperboloidTangent) {
    // a point on the Poincare disc embedded in Minkowski 2+1 space
    poincare::Vector ball_point(3);
    ball_point[0] = 0.1;
    ball_point[1] = -0.2;
    ball_point[2] = 0;

    // a tangent vector in its tangent space
    poincare::Vector ball_tangent(3);
    ball_tangent[0] = -0.1;
    ball_tangent[1] = 1.1;
    ball_tangent[2] = 0.;

    // get the corresponding hyperboloid point
    poincare::Vector hyperboloid_point(ball_point);
    hyperboloid_point.to_hyperboloid_point();

    // get the tangent
    poincare::Vector hyperboloid_tangent(ball_tangent);
    hyperboloid_tangent.to_hyperboloid_tangent(ball_point);
    
    // should be minkowski orthogonal to point
    EXPECT_NEAR(minkowski_dot(hyperboloid_tangent, hyperboloid_point), 0, 1e-8);

    // should be undone by to_ball_tangent
    poincare::Vector tangent(hyperboloid_tangent);
    tangent.to_ball_tangent(hyperboloid_point);
    for (int i=0; i < tangent.dimension_; i++) {
        EXPECT_NEAR(tangent[i], ball_tangent[i], 1e-8);
    } 
}

TEST(VectorTest, geodesicUpdate) {
    // basepoint
    poincare::Vector basepoint(2);
    basepoint[0] = 0.f;
    basepoint[1] = 1.0f;
    // our test point: start out at the basepoint
    poincare::Vector point(basepoint);
    // a tangent vector in its tangent space
    real dist = 3;
    poincare::Vector tangent(2);
    tangent[0] = 1;
    tangent[1] = 0.;
    // apply exponential
    point.geodesic_update(tangent, dist);
    // exponential map is a radial isometry, so should be dist from basepoint
    EXPECT_FLOAT_EQ(dist, distance(basepoint, point));
}

TEST(VectorTest, projectOntoTangentSpace) {
    // basepoint
    poincare::Vector point(2);
    point[0] = 0.;
    point[1] = 1.0;
    poincare::Vector tangent(2);
    tangent[0] = 1.5;
    tangent[1] = 1.0;
    tangent.project_onto_tangent_space(point);
    // tangent should be orthogonal to the vector of the point
    real mdp = minkowski_dot(tangent, point);
    EXPECT_FLOAT_EQ(0., mdp);
}

}    // namespace
