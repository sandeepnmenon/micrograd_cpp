#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <micrograd_cpp/nn.cpp>
using namespace std;
using namespace micrograd_cpp;

vector<VariablePtr> convert_to_variable(vector<float> x)
{
    vector<VariablePtr> variables = vector<VariablePtr>{};
    for (auto i : x)
    {
        variables.emplace_back(make_shared<Variable>(i));
    }

    return variables;
}

void train(MLP model)
{
    // // Entire Training loop
    vector<vector<float>> xs = {
        {2,3,-1},
        {3,-1,0.5},
        {0.5,1,1},
        {1,1,-1}
    };
    vector<float> ys = {1,-1,-1,1};

    float step_size = 0.05;
    for(int k=0;k<100;++k)
    {
        //forward pass
        vector<vector<VariablePtr>> ypred = vector<vector<VariablePtr>>();
        for (vector<float> x : xs)
        {
            vector<VariablePtr> vx = convert_to_variable(x);
            ypred.emplace_back(model(vx));
        }

        //compute loss
        VariablePtr loss = make_shared<Variable>(0.0f);
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
    vector<VariablePtr> x = convert_to_variable(vector<float>{3,1,5});
    auto n = Neuron(3);
    cout<<n<<" => "<<*n(x)<<endl;

    auto layer = Layer(3, 3);
    vector<VariablePtr> output = layer(x);

    for (auto out : output)
    {
        cout<<*out<<endl;
    }
    
    //Define the model
    MLP mlp = MLP(3, {4, 4, 1});
    cout<<mlp<<" with parameters: "<<mlp.parameters().size()<<endl;

    train(mlp);

    return 0;
}
