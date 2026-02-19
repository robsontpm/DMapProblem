#include <iostream>
#include <vector>

namespace ConstMinusVar {
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo)
      result[coeffNo] = -right[coeffNo];
    else
      *result = *left - *right;
  }
}

// Minimal Node infrastructure
class AbstractNode {
public:
    virtual void evalC0(const int coeffNo) = 0;
    virtual ~AbstractNode() {}
};

template<class T>
class ConstMinusVarNode : public AbstractNode {
public:
    const T* left;
    const T* right;
    T* result;

    ConstMinusVarNode(const T* l, const T* r, T* res) : left(l), right(r), result(res) {}

    void evalC0(const int coeffNo) override {
        ConstMinusVar::evalC0(left, right, result, coeffNo);
    }
};

int main() {
    const int order = 5;
    const int num_nodes = 3; // c, y, z
    const int size_per_node = order + 1;

    std::vector<double> memory(num_nodes * size_per_node);

    // Layout:
    // Node 0: c (constant 3)
    // Node 1: y (variable)
    // Node 2: z (result 3-y)

    double* c = &memory[0 * size_per_node];
    double* y = &memory[1 * size_per_node];
    double* z = &memory[2 * size_per_node];

    // Initialize constant
    c[0] = 3.0;
    for(int i=1; i<=order; ++i) c[i] = 0.0;

    // Initialize y_0
    y[0] = 2.0;

    // Node representing z = 3 - y
    ConstMinusVarNode<double> node(c, y, z);

    std::vector<AbstractNode*> nodes;
    nodes.push_back(&node);

    // Solver loop
    for (int k = 0; k <= order; ++k) {
        // Evaluate equation z_k = (3 - y)_k
        for(auto* n : nodes) {
            n->evalC0(k);
        }

        // Integration step: y_{k+1} = z_k / (k+1)
        if (k < order) {
             y[k+1] = z[k] / (k+1);
        }
    }

    for(int i=0; i<=order; ++i) {
        std::cout << "{" << y[i] << "}" << std::endl;
    }

    return 0;
}
