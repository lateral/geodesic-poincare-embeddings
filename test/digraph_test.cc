#include "gtest/gtest.h"
#include "digraph.h"

namespace {

TEST(DigraphTest, TestCreateNode) {
    poincare::Node n("cat", 2);
    EXPECT_EQ(n.name, "cat");
    EXPECT_EQ(n.enumeration, 2);
    EXPECT_EQ(n.count_as_source, 0);
    EXPECT_EQ(n.count_as_target, 0);
    EXPECT_EQ(n.target_enums.size(), 0);
}

TEST(DigraphTest, TestCreateNodeNameIsCopy) {
    std::string name = "cat";
    poincare::Node node(name, 2);
    name[2] = 'r';
    EXPECT_EQ(node.name, "cat");
}

TEST(DigraphTest, TestCreateEdge) {
    poincare::Node n0("cat", 2);
    poincare::Node n1("mammal", 3);
    poincare::Edge edge(n0, n1);
    EXPECT_EQ(edge.target.name, "mammal");
    EXPECT_EQ(n0.count_as_target, 0);
    EXPECT_EQ(n1.count_as_target, 1);
    EXPECT_EQ(n0.target_enums[0], n1.enumeration); 
}

TEST(DigraphTest, TestCreateDigraph) {
    std::string spec = "car\tvehicle\nvehicle\tthing\npotato\tthing\ncat\tmammal\nmammal\tthing";
    std::istringstream in(spec);
    poincare::Digraph dig(in);
    EXPECT_EQ(dig.node_count(), 6);
    EXPECT_EQ(dig.edges.size(), 5);
    poincare::Edge* edge_ptr = dig.edges[0];
    EXPECT_EQ(edge_ptr->source.name, "car");
    EXPECT_EQ(edge_ptr->target.name, "vehicle");
    EXPECT_EQ(dig.node_count(), dig.enumeration2node.size());
    EXPECT_EQ(dig.name2node["thing"]->count_as_target, 3);
    EXPECT_EQ(dig.name2node["thing"]->count_as_source, 0);
    EXPECT_EQ(dig.name2node["car"]->count_as_source, 1);
    EXPECT_EQ(dig.name2node["car"]->target_enums[0], dig.name2node["vehicle"]->enumeration);
}

TEST(DigraphTest, TestCreateDigraphEmpty) {
    std::string spec = "";
    std::istringstream in(spec);
    poincare::Digraph dig(in);
    EXPECT_EQ(dig.node_count(), 0);
    EXPECT_EQ(dig.edges.size(), 0);
}

TEST(DigraphTest, TestCreateDigraphTrailingLinefeed) {
    std::string spec = "car\tvehicle\nvehicle\tthing\npotato\tthing\ncat\tmammal\nmammal\tthing\n";
    std::istringstream in(spec);
    poincare::Digraph dig(in);
    EXPECT_EQ(dig.node_count(), 6);
    EXPECT_EQ(dig.edges.size(), 5);
}

TEST(DigraphTest, TestCreateDigraphTooManyColumns) {
    std::string spec = "car\tvehicle\tvehicle\tthing";
    std::istringstream in(spec);
    try {
        poincare::Digraph dig(in);
        FAIL();
    } catch (std::runtime_error) {
    }
}

}    // namespace
