#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <micrograd_cpp/nn.cpp>
using namespace std;
using namespace micrograd_cpp;

vector<micrograd_cpp::VariablePtr> convert_to_variable(vector<float> x)
{
    vector<VariablePtr> variables = vector<VariablePtr>{};
    for (auto i : x)
    {
        variables.emplace_back(make_shared<Variable>(i));
    }

    return variables;
}

int main(int, char**) {
    std::cout << "Hello, world!\n";
    vector<VariablePtr> x = convert_to_variable(vector<float>{3,1,5});
    auto n = Neuron(3);
    cout<<n<<" => "<<*n(x)<<endl;

    auto layer = Layer(3, 3);
    vector<VariablePtr> output = layer(x);

    for (auto out : output)
    {
        cout<<*out<<endl;
    }

    
    // // Entire Training loop
    // vector<vector<float>> xs = {
    //     {2,3,-1},
    //     {3,-1,0.5},
    //     {0.5,1,1},
    //     {1,1,-1}
    // };
    // vector<float> ys = {1,-1,-1,1};

    // //Define the model
    // auto mlp = MLP(3, {4, 4, 1});
    // cout<<mlp<<" with parameters: "<<mlp.parameters().size()<<endl;

    // float step_size = 0.05;
    // for(int k=0;k<100;++k)
    // {
    //     //forward pass
    //     auto ypred = 
    // }
    
    return 0;
}
