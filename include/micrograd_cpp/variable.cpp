#include <ostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <functional>

using namespace std;

namespace micrograd_cpp
{
    class Variable;
    using VariablePtr = shared_ptr<Variable>;

    class Variable : public enable_shared_from_this<Variable>
    {
        float _data;
        vector<VariablePtr> _prev;
        string op;
        string _label;
        bool topo_visited = false;

        const vector<VariablePtr> &children() const { return _prev; }
        void set_visited() { topo_visited = true; }

        // Overload the + operator
        VariablePtr operator+(const VariablePtr other)
        {
            VariablePtr self = shared_from_this();
            vector<VariablePtr> children = vector<VariablePtr>{self, other};

            VariablePtr out = make_shared<Variable>(self->data() + other->data(), children, "+");
            out->_backward = [out, other, self]()
            {
                self->grad += 1.0 * out->grad;
                other->grad += 1.0 * out->grad;
            };

            return out;
        }

        // Overload the * operator
        VariablePtr operator*(const VariablePtr &other)
        {
            VariablePtr out = make_shared<Variable>(this->data() * other->data(), vector<VariablePtr>{shared_from_this(), other}, "*");
            out->_backward = [out, other, self = shared_from_this()]()
            {
                self->grad += other->data() * out->grad;
                other->grad += self->data() * out->grad;
            };

            return out;
        }

        void build_topo(VariablePtr var, vector<VariablePtr>& topo_order)
        {
            if(!var->topo_visited)
            {
                var->set_visited();
                for(VariablePtr child: var->children())
                {
                    build_topo(child, topo_order);
                }
                topo_order.push_back(var);
            }
        }

        void clear_topo_visited(vector<VariablePtr>& topo_order)
        {
            for(VariablePtr node: topo_order)
            {
                node->topo_visited = false;
            }
        }

    public:
        float grad;
        function<void()> _backward;

        void set_label(string label) { this->_label = label; }
        float data() const { return _data; }
        void set_data(float data) { this->_data = data; }

        
        Variable(float data, vector<VariablePtr> _children = vector<VariablePtr>(), string op = "", string label = "")
        {
            this->_data = data;
            this->grad = 0.0;
            this->_prev = _children;
            this->op = op;
            this->_label = label;
            this->_backward = []() {};
        }

        // function to print label when printing the object using cout
        friend ostream &operator<<(ostream &os, const Variable &v)
        {
            os << "Variable(data=" << v.data() << ", grad=" << v.grad << ", label=" << v._label << ")";
            return os;
        }

        // Define the pow operator
        VariablePtr pow(float exponent)
        {
            VariablePtr self = shared_from_this();
            VariablePtr out = make_shared<Variable>(std::pow(this->data(), exponent), vector<VariablePtr>{self}, "^" + to_string(exponent));
            out->_backward = [out, exponent, self]()
            {
                self->grad += exponent * std::pow(self->data(), exponent - 1) * out->grad;
            };

            return out;
        }

        // Define backpropagation function
        void backward()
        {
            this->grad = 1.0;
            vector<VariablePtr> topo_order;
            build_topo(shared_from_this(), topo_order);
            reverse(topo_order.begin(), topo_order.end());
            for(VariablePtr node: topo_order)
            {
                node->_backward();
            }
            clear_topo_visited(topo_order);
        }

        // Define the tanh function
        VariablePtr tanh()
        {
            float n = this->data();
            float tanh_value = (exp(2 * n) - 1) / (exp(2 * n) + 1);
            VariablePtr out = make_shared<Variable>(tanh_value, vector<VariablePtr>{shared_from_this()}, "tanh");
            out->_backward = [out, self = shared_from_this()]()
            {
                self->grad += (1 - out->data() * out->data()) * out->grad;
            };

            return out;
        }

        // Define the relu function
        VariablePtr relu()
        {
            float n = this->data();
            float relu_value = n > 0 ? n : 0;
            VariablePtr out = make_shared<Variable>(relu_value, vector<VariablePtr>{shared_from_this()}, "relu");
            out->_backward = [out, self = shared_from_this()]()
            {
                self->grad += (out->data() > 0 ? 1 : 0) * out->grad;
            };

            return out;
        }
    };

    inline VariablePtr operator*(VariablePtr lhs, VariablePtr rhs)
    {
        VariablePtr out = make_shared<Variable>(lhs->data() * rhs->data(), vector<VariablePtr>{lhs, rhs}, "*");
        out->_backward = [out, lhs, rhs]()
        {
            lhs->grad += rhs->data() * out->grad;
            rhs->grad += lhs->data() * out->grad;
        };

        return out;
    }

    inline VariablePtr operator*(VariablePtr lhs, float val)
    {
        return lhs * make_shared<Variable>(val);
    }

    inline VariablePtr operator*(float val, VariablePtr rhs)
    {
        return make_shared<Variable>(val) * rhs;
    }

    inline VariablePtr operator+(VariablePtr lhs, VariablePtr rhs)
    {
        VariablePtr out = make_shared<Variable>(lhs->data() + rhs->data(), vector<VariablePtr>{lhs, rhs}, "+");
        out->_backward = [out, lhs, rhs]()
        {
            lhs->grad += out->grad;
            rhs->grad += out->grad;
        };
        return out;
    }

    inline VariablePtr operator+(VariablePtr lhs, float val)
    {
        return lhs + make_shared<Variable>(val);
    }

    inline VariablePtr operator+(float val, VariablePtr rhs)
    {
        return make_shared<Variable>(val) + rhs;
    }

    inline VariablePtr operator-(VariablePtr rhs)
    {
        return rhs * make_shared<Variable>(-1.0);
    }

    inline VariablePtr operator-(VariablePtr lhs, VariablePtr rhs)
    {
        return lhs + (-rhs);
    }

    inline VariablePtr operator-(VariablePtr lhs, float val)
    {
        return lhs + (-make_shared<Variable>(val));
    }

    inline VariablePtr operator-(float val, VariablePtr rhs)
    {
        return make_shared<Variable>(val) + (-rhs);
    }

    inline VariablePtr operator/(VariablePtr lhs, VariablePtr rhs)
    {
        return lhs * rhs->pow(-1);
    }

    inline VariablePtr operator/(VariablePtr lhs, float val)
    {
        return lhs / make_shared<Variable>(val);
    }

    inline VariablePtr operator/(float val, VariablePtr rhs)
    {
        return make_shared<Variable>(val) / rhs;
    }
}
