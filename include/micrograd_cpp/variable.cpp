#ifndef __MICROGRAD_CPP_VARIABLE_CPP_
#define __MICROGRAD_CPP_VARIABLE_CPP_

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
    template <typename T> 
    class Variable;

    template <typename T> 
    using VariablePtr = shared_ptr<Variable<T>>;

    template <typename T> 
    class Variable :public enable_shared_from_this<Variable<T>>
    {
        T _data;
        vector<VariablePtr<T>> _prev;
        string op;
        string _label;
        bool topo_visited = false;

        const vector<VariablePtr<T>> &children() const { return _prev; }
        void set_visited() { topo_visited = true; }

        // Overload the + operator
        VariablePtr<T> operator+(const VariablePtr<T> other)
        {
            VariablePtr<T> self = enable_shared_from_this<Variable<T>>::shared_from_this();
            vector<VariablePtr<T>> children = vector<VariablePtr<T>>{self, other};

            VariablePtr<T> out = make_shared<Variable>(self->data() + other->data(), children, "+");
            out->_backward = [out, other, self]()
            {
                self->grad += 1.0 * out->grad;
                other->grad += 1.0 * out->grad;
            };

            return out;
        }

        // Overload the * operator
        VariablePtr<T> operator*(const VariablePtr<T> &other)
        {
            VariablePtr<T> out = make_shared<Variable>(this->data() * other->data(), vector<VariablePtr<T>>{enable_shared_from_this<Variable<T>>::shared_from_this(), other}, "*");
            out->_backward = [out, other, self = enable_shared_from_this<Variable<T>>::shared_from_this()]()
            {
                self->grad += other->data() * out->grad;
                other->grad += self->data() * out->grad;
            };

            return out;
        }

        void build_topo(VariablePtr<T> var, vector<VariablePtr<T>>& topo_order)
        {
            if(!var->topo_visited)
            {
                var->set_visited();
                for(VariablePtr<T> child: var->children())
                {
                    build_topo(child, topo_order);
                }
                topo_order.push_back(var);
            }
        }

        void clear_topo_visited(vector<VariablePtr<T>>& topo_order)
        {
            for(VariablePtr<T> node: topo_order)
            {
                node->topo_visited = false;
            }
        }

    public:
        T grad;
        function<void()> _backward;

        void set_label(string label) { this->_label = label; }
        T data() const { return _data; }
        void set_data(T data) { this->_data = data; }

        
        Variable(T data, vector<VariablePtr<T>> _children = vector<VariablePtr<T>>(), string op = "", string label = "")
        {
            this->_data = data;
            this->grad = 0.0;
            this->_prev = _children;
            this->op = op;
            this->_label = label;
            this->_backward = []() {};
        }

        static VariablePtr<T> makeVariable(T data)
        {
            return make_shared<Variable<T>>(data);
        }

        // function to print label when printing the object using cout
        friend ostream &operator<<(ostream &os, const Variable &v)
        {
            os << "Variable(data=" << v.data() << ", grad=" << v.grad << ", label=" << v._label << ")";
            return os;
        }

        // Define the pow operator
        VariablePtr<T> pow(float exponent)
        {
            VariablePtr<T> self = enable_shared_from_this<Variable<T>>::shared_from_this();
            VariablePtr<T> out = make_shared<Variable>(std::pow(this->data(), exponent), vector<VariablePtr<T>>{self}, "^" + to_string(exponent));
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
            vector<VariablePtr<T>> topo_order;
            build_topo(enable_shared_from_this<Variable<T>>::shared_from_this(), topo_order);
            reverse(topo_order.begin(), topo_order.end());
            for(VariablePtr<T> node: topo_order)
            {
                node->_backward();
            }
            clear_topo_visited(topo_order);
        }

        // Define the tanh function
        VariablePtr<T> tanh()
        {
            T n = this->data();
            T tanh_value = (exp(2 * n) - 1) / (exp(2 * n) + 1);
            VariablePtr<T> out = make_shared<Variable>(tanh_value, vector<VariablePtr<T>>{enable_shared_from_this<Variable<T>>::shared_from_this()}, "tanh");
            out->_backward = [out, self = enable_shared_from_this<Variable<T>>::shared_from_this()]()
            {
                self->grad += (1 - out->data() * out->data()) * out->grad;
            };

            return out;
        }

        // Define the relu function
        VariablePtr<T> relu()
        {
            T n = this->data();
            T relu_value = n > 0 ? n : 0;
            VariablePtr<T> out = make_shared<Variable>(relu_value, vector<VariablePtr<T>>{enable_shared_from_this<Variable<T>>::shared_from_this()}, "relu");
            out->_backward = [out, self = enable_shared_from_this<Variable<T>>::shared_from_this()]()
            {
                self->grad += (out->data() > 0 ? 1 : 0) * out->grad;
            };

            return out;
        }
    };

    template <typename T> inline VariablePtr<T> operator*(VariablePtr<T> lhs, VariablePtr<T> rhs)
    {
        VariablePtr<T> out = make_shared<Variable<T>>(lhs->data() * rhs->data(), vector<VariablePtr<T>>{lhs, rhs}, "*");
        out->_backward = [out, lhs, rhs]()
        {
            lhs->grad += rhs->data() * out->grad;
            rhs->grad += lhs->data() * out->grad;
        };

        return out;
    }

    template <typename T> inline VariablePtr<T> operator*(VariablePtr<T> lhs, T val)
    {
        return lhs * make_shared<Variable>(val);
    }

    // template <typename T> inline VariablePtr<T> operator*(T val, VariablePtr<T> rhs)
    // {
    //     return make_shared<Variable>(val) * rhs;
    // }

    template <typename T> inline VariablePtr<T> operator+(VariablePtr<T> lhs, VariablePtr<T> rhs)
    {
        VariablePtr<T> out = make_shared<Variable<T>>(lhs->data() + rhs->data(), vector<VariablePtr<T>>{lhs, rhs}, "+");
        out->_backward = [out, lhs, rhs]()
        {
            lhs->grad += out->grad;
            rhs->grad += out->grad;
        };
        return out;
    }

    template <typename T> inline VariablePtr<T> operator+(VariablePtr<T> lhs, T val)
    {
        return lhs + make_shared<Variable>(val);
    }

    template <typename T> inline VariablePtr<T> operator+(T val, VariablePtr<T> rhs)
    {
        return make_shared<Variable>(val) + rhs;
    }

    template <typename T> inline VariablePtr<T> operator-(VariablePtr<T> rhs)
    {
        return rhs * make_shared<Variable<T>>(-1.0);
    }

    template <typename T> inline VariablePtr<T> operator-(VariablePtr<T> lhs, VariablePtr<T> rhs)
    {
        return lhs + (-rhs);
    }

    template <typename T> inline VariablePtr<T> operator-(VariablePtr<T> lhs, T val)
    {
        return lhs + (-make_shared<Variable<T>>(val));
    }

    template <typename T> inline VariablePtr<T> operator-(T val, VariablePtr<T> rhs)
    {
        return make_shared<Variable<T>>(val) + (-rhs);
    }

    template <typename T> inline VariablePtr<T> operator/(VariablePtr<T> lhs, VariablePtr<T> rhs)
    {
        return lhs * rhs->pow(-1);
    }

    template <typename T> inline VariablePtr<T> operator/(VariablePtr<T> lhs, T val)
    {
        return lhs / make_shared<Variable<T>>(val);
    }

    template <typename T> inline VariablePtr<T> operator/(T val, VariablePtr<T> rhs)
    {
        return make_shared<Variable<T>>(val) / rhs;
    }
}

#endif