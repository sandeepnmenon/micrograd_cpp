# micrograd_cpp
C++ implementaton of the scalar-valued autograd engine and a simple neural net library built using the same


## Build
Very simple setup procedure as in a normal CMake project.

```
mkdir build
cd build
cmake ..
make
```

## Building the computational graph
The computational graph is built by creating a `Variable` object and performing operations on it. The operations are overloaded to return a new `Variable` object. The `Variable` object has a `grad` attribute which is the gradient of the variable with respect to the loss function. The `Variable` object also has a `backward` method which computes the gradient of the variable with respect to the loss function. The `backward` method is called on the loss variable and the gradients are computed recursively.

For example to build the computations of a neuron
```cpp
#include <micrograd_cpp/variable.cpp>
using namespace micrograd;

int main()
{
    // Sample input variables
    auto x1 = Variable<float>::makeVariable(2.0f);
    auto x2 = Variable<float>::makeVariable(1.0f);

    // Sample weights and bias
    auto w1 = Variable<float>::makeVariable(-3.0f);
    auto w2 = Variable<float>::makeVariable(0.0f);
    auto b = Variable<float>::makeVariable(8.0f);

    // compute the neuron operations 
    //output = tanh(x1*w1 + x2*w2 + b)
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto neuron1=  x1w1x2w2 + b;
    auto output = neuron1->tanh();

    // compute gradients via backpropagation
    output->backward();

    // print gradients
    cout<<*w1<<endl;
    cout<<*w2<<endl;
    
    return 0;
}
```
Note: the above example is not comprehensive and is just to give an idea of how to use the library.

## Creating a neural net
The neural net library is built using the same `Variable` class. Class `MLP` can be used to define a basic fully connected neural network. The `backward` method is called on the loss variable and the gradients are computed recursively.

For example to build a neural net
```cpp
#include <micrograd_cpp/nn.cpp>
#include <vector>
using namespace std;
using namespace micrograd;

int main()
{
    MLP<float> model = MLP<float>(3, {4, 4, 1});
    float step_size = 0.01f;
    //forward pass
    vector<vector<VariablePtr<T>>> ypred = vector<vector<VariablePtr<T>>>();
    for (vector<float> x : xs)  // xs is the input data
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

    //set zero gradients
    model.zero_grad();

    //backward pass
    loss->backward();

    //update parameters
    for(auto p : model.parameters())
    {
        p->set_data(p->data() - step_size * p->grad);
    }

    return 0;
}

```
Note: the above example is not comprehensive and is just to give an idea of how to use the library.

## TODO
- [ ] Add support for non updatable parameters
- [ ] Wrapper class to pass in the datatype as an argument instead of having to change the template argument everywhere