#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// --- Helpers ---
inline int binomial(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    long res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

// --- DagIndexer Stub ---
template<class T>
class DagIndexer {
public:
    int domainDimension() const { return 1; }
    int getOrder() const { return 5; }
};

// --- AbstractNode Template ---
template<class T>
class AbstractNode {
public:
    T* left;
    T* right;
    T* result;
    DagIndexer<T>* dag;

    AbstractNode() : left(nullptr), right(nullptr), result(nullptr), dag(nullptr) {}

    void setDag(DagIndexer<T>* d) { dag = d; }

    virtual void evalC0(const int coeffNo) = 0;
    // Add eval to match library
    virtual void eval(const int degree, const int coeffNo) = 0;
    virtual ~AbstractNode() {}
};

// --- ConstMinusVar Namespace ---
namespace ConstMinusVar {
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo)
      result[coeffNo] = -right[coeffNo];
    else
      *result = *left - *right;
  }

  template<class T, class R>
  inline void evalHelper(const T* right, R result, const unsigned dataSize, const unsigned order, const unsigned shift)
  {
    // Dummy implementation of evalHelper
    right += shift;
    result += shift;
    const T* end = right + dataSize*order;
    for(;right!=end; right+=order,result+=order)
        *result = -(*right);
  }

  template<class T, class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    evalHelper(right,result,binomial(dim+degree,degree)-1,order,coeffNo+order);
  }
}

// --- Macros for Dummy Nodes ---
#define MAKE_DUMMY_NODE(NAME) \
namespace NAME { \
  template<class T, class R> \
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo) { \
    result[coeffNo] = left[coeffNo] + right[coeffNo]; \
  } \
  template<class T, class R> \
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo) { \
     evalC0(left, right, result, coeffNo); \
  } \
} \
template<class T> \
class NAME##Node : public AbstractNode<T> { \
public: \
    void evalC0(const int coeffNo) override { \
        NAME::evalC0(this->left, this->right, this->result, coeffNo); \
    } \
    void eval(const int degree, const int coeffNo) override { \
        NAME::eval(degree, this->left, this->right, this->result, this->dag, coeffNo); \
    } \
};

// Create many dummy nodes
MAKE_DUMMY_NODE(Add)
MAKE_DUMMY_NODE(Sub)
MAKE_DUMMY_NODE(Mul)
MAKE_DUMMY_NODE(Div)
MAKE_DUMMY_NODE(Sqr)
MAKE_DUMMY_NODE(Sqrt)
MAKE_DUMMY_NODE(Exp)
MAKE_DUMMY_NODE(Log)
MAKE_DUMMY_NODE(Sin)
MAKE_DUMMY_NODE(Cos)
MAKE_DUMMY_NODE(Atan)
MAKE_DUMMY_NODE(Asin)
MAKE_DUMMY_NODE(Acos)
MAKE_DUMMY_NODE(Dummy1)
MAKE_DUMMY_NODE(Dummy2)
MAKE_DUMMY_NODE(Dummy3)
MAKE_DUMMY_NODE(Dummy4)
MAKE_DUMMY_NODE(Dummy5)
MAKE_DUMMY_NODE(Dummy6)
MAKE_DUMMY_NODE(Dummy7)
MAKE_DUMMY_NODE(Dummy8)
MAKE_DUMMY_NODE(Dummy9)
MAKE_DUMMY_NODE(Dummy10)

// --- ConstMinusVarNode ---
template<class T>
class ConstMinusVarNode : public AbstractNode<T> {
public:
    void evalC0(const int coeffNo) override {
        ConstMinusVar::evalC0(this->left, this->right, this->result, coeffNo);
    }
    void eval(const int degree, const int coeffNo) override {
        ConstMinusVar::eval(degree, this->left, this->right, this->result, this->dag, coeffNo);
    }
};

enum NodeType {
    NODE_ADD, NODE_SUB, NODE_MUL, NODE_DIV,
    NODE_SQR, NODE_SQRT, NODE_EXP, NODE_LOG,
    NODE_SIN, NODE_COS, NODE_ATAN, NODE_ASIN, NODE_ACOS,
    NODE_CONST_MINUS_VAR,
    NODE_DUMMY1, NODE_DUMMY2, NODE_DUMMY3, NODE_DUMMY4, NODE_DUMMY5,
    NODE_DUMMY6, NODE_DUMMY7, NODE_DUMMY8, NODE_DUMMY9, NODE_DUMMY10
};

template<class T>
AbstractNode<T>* createNode(NodeType type, T* left, T* right, T* result, DagIndexer<T>* dag) {
    AbstractNode<T>* p = nullptr;
    switch(type) {
        case NODE_ADD: p = new AddNode<T>(); break;
        case NODE_SUB: p = new SubNode<T>(); break;
        case NODE_MUL: p = new MulNode<T>(); break;
        case NODE_DIV: p = new DivNode<T>(); break;
        case NODE_SQR: p = new SqrNode<T>(); break;
        case NODE_SQRT: p = new SqrtNode<T>(); break;
        case NODE_EXP: p = new ExpNode<T>(); break;
        case NODE_LOG: p = new LogNode<T>(); break;
        case NODE_SIN: p = new SinNode<T>(); break;
        case NODE_COS: p = new CosNode<T>(); break;
        case NODE_ATAN: p = new AtanNode<T>(); break;
        case NODE_ASIN: p = new AsinNode<T>(); break;
        case NODE_ACOS: p = new AcosNode<T>(); break;
        case NODE_CONST_MINUS_VAR: p = new ConstMinusVarNode<T>(); break;
        case NODE_DUMMY1: p = new Dummy1Node<T>(); break;
        case NODE_DUMMY2: p = new Dummy2Node<T>(); break;
        case NODE_DUMMY3: p = new Dummy3Node<T>(); break;
        case NODE_DUMMY4: p = new Dummy4Node<T>(); break;
        case NODE_DUMMY5: p = new Dummy5Node<T>(); break;
        case NODE_DUMMY6: p = new Dummy6Node<T>(); break;
        case NODE_DUMMY7: p = new Dummy7Node<T>(); break;
        case NODE_DUMMY8: p = new Dummy8Node<T>(); break;
        case NODE_DUMMY9: p = new Dummy9Node<T>(); break;
        case NODE_DUMMY10: p = new Dummy10Node<T>(); break;
        default: throw std::runtime_error("Unknown node type");
    }
    p->left = left;
    p->right = right;
    p->result = result;
    p->setDag(dag);
    return p;
}

// --- Main ---

int main() {
    const int order = 5;
    const int num_nodes = 50;
    const int size_per_node = order + 1;

    std::vector<double> memory(num_nodes * size_per_node);
    for(size_t i=0; i<memory.size(); ++i) memory[i] = 0.0;

    double* c = &memory[0 * size_per_node];
    double* y = &memory[1 * size_per_node];
    double* z = &memory[2 * size_per_node]; // z = 3 - y

    c[0] = 3.0;
    y[0] = 2.0;

    DagIndexer<double> dag;

    std::vector<AbstractNode<double>*> nodes;

    // Add some noise nodes
    for(int i=0; i<10; ++i) {
         nodes.push_back(createNode(NODE_ADD, c, c, &memory[3*size_per_node], &dag));
    }

    // The target node
    nodes.push_back(createNode(NODE_CONST_MINUS_VAR, c, y, z, &dag));

    // Add more noise nodes
    for(int i=0; i<10; ++i) {
         nodes.push_back(createNode(NODE_MUL, c, c, &memory[4*size_per_node], &dag));
    }

    // Solver loop
    for (int k = 0; k <= order; ++k) {
        // Evaluate equations
        for(auto* n : nodes) {
            n->evalC0(k);
        }

        if (k < order) {
             y[k+1] = z[k] / (k+1);
        }
    }

    for(int i=0; i<=order; ++i) {
        std::cout << "{" << y[i] << "}" << std::endl;
    }

    // cleanup
    for(auto* n : nodes) delete n;

    return 0;
}
