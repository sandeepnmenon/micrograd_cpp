#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <micrograd_cpp/nn.cpp>
using namespace std;
using namespace micrograd_cpp;

template <typename T>
vector<VariablePtr<T>> convert_to_variable(vector<T> x)
{
    vector<VariablePtr<T>> variables = vector<VariablePtr<T>>{};
    for (T i : x)
    {
        variables.emplace_back(Variable<T>::makeVariable(i));
    }

    return variables;
}

template <typename T>
void train(MLP<T> model)
{
    // // Entire Training loop
    vector<vector<T>> xs = {
        {2,3,-1},
        {3,-1,0.5},
        {0.5,1,1},
        {1,1,-1}
    };
    vector<T> ys = {1,-1,-1,1};

    float step_size = 0.05;
    for(int k=0;k<100;++k)
    {
        //forward pass
        vector<vector<VariablePtr<T>>> ypred = vector<vector<VariablePtr<T>>>();
        for (vector<float> x : xs)
        {
            vector<VariablePtr<T>> vx = convert_to_variable<T>(x);
            ypred.emplace_back(model(vx));
        }

        //compute loss
        VariablePtr<T> loss = Variable<T>::makeVariable(0.0f);
        for(int i = 0; i < ypred.size(); ++i)
        {
            loss = loss + (ypred[i][0] - ys[i]) * (ypred[i][0] - ys[i]);
        }

        model.zero_grad();

        //backward pass
        loss->backward();

        //update parameters
        for(auto p : model.parameters())
        {
            p->set_data(p->data() - step_size * p->grad);
        }

        if(k%10==0)
            cout<<k<<": Loss: "<<*loss<<endl;
    }
}

int main(int, char**) {
    std::cout << "Hello, world!\n";
    vector<VariablePtr<float>> x = convert_to_variable<float>(vector<float>{3,1,5});
    auto n = Neuron<float>(3);
    cout<<n<<" => "<<*n(x)<<endl;

    auto layer = Layer<float>(3, 3);
    vector<VariablePtr<float>> output = layer(x);

    for (auto out : output)
    {
        cout<<*out<<endl;
    }
    
    //Define the model
    MLP<float> mlp = MLP<float>(3, {4, 4, 1});
    cout<<mlp<<" with parameters: "<<mlp.parameters().size()<<endl;

    train<float>(mlp);

    return 0;
}
