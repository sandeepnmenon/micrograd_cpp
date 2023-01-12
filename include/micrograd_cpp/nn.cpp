#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <cassert>

#include <micrograd_cpp/variable.cpp>

namespace micrograd_cpp
{
    enum Activation
    {
        TANH,
        RELU
    };
    
    static std::unordered_map<std::string,Activation> const activation_table = { {"tanh",Activation::TANH}, {"relu",Activation::RELU} };

    class Module
    {
        void zero_grad()
        {
            for(VariablePtr param: parameters())
            {
                param->grad = 0.0f;
            }
        }

        vector<VariablePtr> parameters()
        {
            return vector<VariablePtr>{};
        }
    };

    class Neuron: Module
    {
        vector<VariablePtr> _weights;
        VariablePtr _bias;
        Activation _activation;
        string activation_name;

        void initialize_weights_uniform(unsigned int nin)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distr(-1.0, 1.0);

            this->_bias->set_data(distr(gen));

            for(unsigned int i = 0; i < nin; i++)
            {
                this->_weights.emplace_back(make_shared<Variable>(distr(gen)));
            }

        }

    public:
        Neuron(unsigned int nin, string activation="tanh")
        {
            this->_activation = activation_table.at(activation);
            this->activation_name = activation;
            this->_weights = vector<VariablePtr>{};
            this->_bias = make_shared<Variable>(0.0f);
            
            initialize_weights_uniform(nin);
        }

        friend ostream& operator<<(ostream &os, const Neuron& n)
        {
            os<<"Neuron("<<n._weights.size()<<", "<<n.activation_name<<")";
            return os;
        }

        VariablePtr operator()(vector<VariablePtr> x)
        {
            auto activation = this->_bias;
            for(unsigned int i = 0; i < x.size(); i++)
            {
                activation = activation + this->_weights[i] * x[i];
            }

            switch (this->_activation)
            {
            case Activation::TANH:
                activation = activation->tanh();
                break;
            case Activation::RELU:
                activation = activation->relu();
                break;
            default:
                // Throw invalid value exception
                throw std::invalid_argument("Invalid activation function");
                break;
            }

            return activation;
        }

        vector<VariablePtr> parameters()
        {
            vector<VariablePtr> parameters = this->_weights;
            parameters.emplace_back(this->_bias);

            return parameters;
        }

    };

    class Layer: Module
    {
        unsigned int nin;
        unsigned int nout;
        vector<Neuron> _neurons;

    public:
        Layer(unsigned int nin, unsigned int nout, string activation="tanh")
        {
            this->nin = nin;
            this->nout = nout;
            this->_neurons = vector<Neuron>{};
            for(unsigned int i = 0; i < nout; i++)
            {
                this->_neurons.emplace_back(Neuron(nin, activation));
            }
        }

        vector<VariablePtr> operator()(vector<VariablePtr> x)
        {
            vector<VariablePtr> layer_output = vector<VariablePtr>{};
            for(Neuron neuron: this->_neurons)
            {
                layer_output.emplace_back(neuron(x));
            }

            return layer_output;
        }

        vector<VariablePtr> parameters()
        {
            vector<VariablePtr> parameters = vector<VariablePtr>{};
            for(Neuron neuron: this->_neurons)
            {
                for(VariablePtr param: neuron.parameters())
                {
                    parameters.emplace_back(param);
                }
            }

            return parameters;
        }

        friend ostream& operator<<(ostream &os, const Layer& l)
        {
            os<<"Layer("<<l.nin<<", "<<l.nout<<")";
            return os;
        }
    };

    class MLP: Module
    {
        unsigned int nin;
        vector<unsigned int> nouts;
        vector<Layer> _layers;

    public:
        MLP(unsigned int nin, vector<unsigned int> nouts, vector<string> activations=vector<string>{})
        {
            if(activations.size() == 0)
            {
                activations = vector<string>(nouts.size(), "tanh");
            }
            else
            {
                assert(activations.size() == nouts.size());
            }
            this->nin = nin;
            this->nouts = nouts;
            this->_layers = vector<Layer>{};
            vector<unsigned int> layer_sizes = vector<unsigned int>{nin};
            layer_sizes.insert(layer_sizes.end(), nouts.begin(), nouts.end());
            for(unsigned int i = 0; i < nouts.size(); i++)
            {
                this->_layers.emplace_back(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]));
            }
        }

        vector<VariablePtr> operator()(vector<VariablePtr> x)
        {
            vector<VariablePtr> layer_output = x;
            for(Layer layer: this->_layers)
            {
                layer_output = layer(layer_output);
            }

            return layer_output;
        }

        vector<VariablePtr> parameters()
        {
            vector<VariablePtr> parameters = vector<VariablePtr>{};
            for(Layer layer: this->_layers)
            {
                for(VariablePtr param: layer.parameters())
                {
                    parameters.emplace_back(param);
                }
            }

            return parameters;
        }

        friend ostream& operator<<(ostream &os, const MLP& m)
        {
            os<<"MLP("<<m.nin<<", [";
            for(unsigned int i = 0; i < m.nouts.size(); i++)
            {
                os<<m.nouts[i];
                if(i < m.nouts.size() - 1)
                {
                    os<<", ";
                }
            }
            os<<"])";
            return os;
        }
    };
}
