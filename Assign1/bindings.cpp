#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essential for converting std::vector to Python lists
#include "Assig1_CNN_model.cpp" 

namespace py = pybind11;

PYBIND11_MODULE(cnn_engine, m) {
    m.doc() = "C++ CNN Engine for Python";

    // --- Bind ConvLayer ---
    py::class_<ConvLayer>(m, "ConvLayer")
        .def(py::init<int>())
        .def_readwrite("filter", &ConvLayer::filter)
        .def_readwrite("bias", &ConvLayer::bias)
        .def_readwrite("k_size", &ConvLayer::k_size)
        .def_readwrite("input", &ConvLayer::input)
        .def_readwrite("grad_weights", &ConvLayer::grad_weights)
        .def_readwrite("bias_grad", &ConvLayer::bias_grad)
        .def("forward", &ConvLayer::forward)
        .def("backward", &ConvLayer::backward) // Signature: (grad, lr)
        .def("getParams", &ConvLayer::getParams)
        .def("getMACs", &ConvLayer::getMACs);

    
        py::class_<FCLayer>(m, "FCLayer")
        .def(py::init<int, int>()) // Handles FCLayer(in_size, out_size)
        .def_readwrite("weights", &FCLayer::weights)   // vector<vector<double>>
        .def_readwrite("bias", &FCLayer::bias)         // vector<double>
        .def_readwrite("input", &FCLayer::input)       // vector<double> (Used for backward)
        .def_readwrite("in_size", &FCLayer::in_size)   // int
        .def_readwrite("out_size", &FCLayer::out_size) // int
        .def("forward", &FCLayer::forward)             // vector<double> forward(vector<double>)
        .def("backward", &FCLayer::backward)           // vector<double> backward(vector<double>, double)
        .def("getParams", &FCLayer::getParams)
        .def("getMACs", &FCLayer::getMACs);

    // --- Bind FlattenLayer ---
    py::class_<FlattenLayer>(m, "FlattenLayer")
        .def(py::init<>())
        // Exposing these allows Python to see the dimensions cached during forward()
        .def_readwrite("in_h", &FlattenLayer::in_h)
        .def_readwrite("in_w", &FlattenLayer::in_w)
        .def("forward", &FlattenLayer::forward)
        .def("backward", &FlattenLayer::backward);

    // --- Bind MaxPoolLayer ---
    py::class_<MaxPoolLayer>(m, "MaxPoolLayer")
        .def(py::init<>())
        // Expose data members
        .def_readwrite("input", &MaxPoolLayer::input)   // vector<vector<double>>
        .def_readwrite("stride", &MaxPoolLayer::stride) // int
        // Expose methods
        .def("forward", &MaxPoolLayer::forward)
        .def("backward", &MaxPoolLayer::backward);

    // --- Bind ReLULayer ---
    py::class_<ReLULayer>(m, "ReLULayer")
        .def(py::init<>())
        // Expose the cached input so Python can see what the ReLU is masking
        .def_readwrite("input", &ReLULayer::input) 
        .def("forward", &ReLULayer::forward)
        .def("backward", &ReLULayer::backward);

    // --- Bind SigmoidLayer ---
    py::class_<SigmoidLayer>(m, "SigmoidLayer")
        .def(py::init<>())
        // Expose the cached output (probabilities)
        .def_readwrite("output", &SigmoidLayer::output) 
        .def("forward", &SigmoidLayer::forward)
        .def("backward", &SigmoidLayer::backward);

    // --- Bind Loss ---
    py::class_<Loss>(m, "Loss")
        .def(py::init<>())
        .def("calculate", &Loss::calculate)
        .def("backward", &Loss::backward);
        
 }