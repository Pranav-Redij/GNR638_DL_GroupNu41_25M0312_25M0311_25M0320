# CNN Model – Train & Test Guide

---
{Dataset 1 and 2 is trained and tested differently and model parameter's stored differently}
## 0. Create `.so` File

### Required Files

```
CNN_model_layers.cpp  → contains layer implementations  
bindings.cpp          → contains pybind11 binding code
Assig1_train.py → contains frontend  
model_digits.pkl, model_objects.pkl → stored parameters of trained model 
```

---

## 0.1 Install pybind11

```bash
pip install pybind11
```

---

## 0.2 Run Binding Command

### ▶ If using Python 3.13 (Mac – MacBook)

```bash
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) bindings.cpp -o cnn_engine.cpython-313-darwin.so -undefined dynamic_lookup
```

### ▶ If using Python 3.10

```bash
g++ -O3 -Wall -shared -std=c++11 -fPIC $(/usr/local/bin/python3 -m pybind11 --includes) bindings.cpp -o cnn_engine.cpython-310-darwin.so -undefined dynamic_lookup
```

This command will create the `.so` file, which contains the layer classes.

---

## Initialize Model



```
# Run 1st Cell: Initialize Model to Be used for Training and Testing
```

---

# 1. Train Model

- Run `loaddataset` cell  
  (Data from both paths will be loaded)

- Then run `train dataset` cell  
  (Change dataset path if needed)

---

# 2. Test Model

- Just run `test model` cell
(change dataset path , like dataset1 or dataset2)
---

