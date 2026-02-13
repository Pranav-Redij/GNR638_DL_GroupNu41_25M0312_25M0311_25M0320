#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "CNN_model_layers.cpp"

namespace py = pybind11;

PYBIND11_MODULE(cnn_engine, m) {
    m.doc() = "C++ CNN Engine (Multi-Channel Version)";

    py::class_<ReLU1DLayer>(m, "ReLU1DLayer")
    .def(py::init<>())
    .def_readwrite("input", &ReLU1DLayer::input)
    .def("forward", &ReLU1DLayer::forward)
    .def("backward", &ReLU1DLayer::backward);


    // ---------------- ConvLayer ----------------
    py::class_<ConvLayer>(m, "ConvLayer")
        .def(py::init<int, int, int>())   // (kernel_size, in_channels, num_filters)
        .def_readwrite("filters", &ConvLayer::filters)
        .def_readwrite("biases", &ConvLayer::biases)
        .def_readwrite("k_size", &ConvLayer::k_size)
        .def_readwrite("in_channels", &ConvLayer::in_channels)
        .def_readwrite("num_filters", &ConvLayer::num_filters)
        .def_readwrite("input", &ConvLayer::input)
        .def("forward", &ConvLayer::forward)
        .def("backward", &ConvLayer::backward)
        .def("getParams", &ConvLayer::getParams)
        .def("getMACs", &ConvLayer::getMACs);


    // ---------------- Fully Connected ----------------
    py::class_<FCLayer>(m, "FCLayer")
        .def(py::init<int, int>())
        .def_readwrite("weights", &FCLayer::weights)
        .def_readwrite("bias", &FCLayer::bias)
        .def_readwrite("input", &FCLayer::input)
        .def_readwrite("in_size", &FCLayer::in_size)
        .def_readwrite("out_size", &FCLayer::out_size)
        .def("forward", &FCLayer::forward)
        .def("backward", &FCLayer::backward)
        .def("getParams", &FCLayer::getParams)
        .def("getMACs", &FCLayer::getMACs);

    // ---------------- Flatten ----------------
    py::class_<FlattenLayer>(m, "FlattenLayer")
        .def(py::init<>())
        .def_readwrite("in_c", &FlattenLayer::in_c)
        .def_readwrite("in_h", &FlattenLayer::in_h)
        .def_readwrite("in_w", &FlattenLayer::in_w)
        .def("forward", &FlattenLayer::forward)     // 3D → 1D
        .def("backward", &FlattenLayer::backward);  // 1D → 3D

    // ---------------- MaxPool ----------------
    py::class_<MaxPoolLayer>(m, "MaxPoolLayer")
        .def(py::init<>())
        .def_readwrite("input", &MaxPoolLayer::input)
        .def_readwrite("stride", &MaxPoolLayer::stride)
        .def("forward", &MaxPoolLayer::forward)     // 3D → 3D
        .def("backward", &MaxPoolLayer::backward);  // 3D → 3D

    // ---------------- ReLU ----------------
    py::class_<ReLULayer>(m, "ReLULayer")
        .def(py::init<>())
        .def_readwrite("input", &ReLULayer::input)
        .def("forward", &ReLULayer::forward)     // 3D → 3D
        .def("backward", &ReLULayer::backward);  // 3D → 3D

    // ---------------- Loss (Softmax + CrossEntropy) ----------------
    py::class_<Loss>(m, "Loss")
        .def(py::init<>())
        .def("softmax", &Loss::softmax)
        .def("calculate", &Loss::calculate)
        .def("backward", &Loss::backward);
}
