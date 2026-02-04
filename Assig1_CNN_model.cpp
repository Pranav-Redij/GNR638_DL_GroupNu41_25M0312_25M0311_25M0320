#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// DONE--- Convolution Layer (2D) ---
class ConvLayer {
    public:
        vector<vector<double>> filter; 
        double bias;
        int k_size;

        vector<vector<double>> input;         
        vector<vector<double>> grad_weights;  
        double bias_grad;                  

        ConvLayer(int k_size) {
            this->k_size = k_size;
            this->bias = ((rand() % 2000) / 10000.0) - 0.1;
            this->filter.assign(k_size, vector<double>(k_size));
            for (int i = 0; i < k_size; i++) {
                for (int j = 0; j < k_size; j++) {
                    this->filter[i][j] = ((rand() % 2000) / 10000.0) - 0.1;
                }
            }
        }

        vector<vector<double>> forward(const vector<vector<double>>& x) {
            // IMPORTANT: Save input for backward pass
            this->input = x; 

            int in_h = x.size();
            int in_w = x[0].size();
            int out_h = in_h - k_size + 1;
            int out_w = in_w - k_size + 1;

            vector<vector<double>> output(out_h, vector<double>(out_w, 0.0));

            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    double sum = bias;//adding bias
                    for (int ki = 0; ki < k_size; ki++) {
                        for (int kj = 0; kj < k_size; kj++) {
                            sum += x[i + ki][j + kj] * filter[ki][kj];
                        }
                    }
                    output[i][j] = sum;
                }
            }
            return output; 
    }

    vector<vector<double>> backward(const vector<vector<double>>& grad_output, double learning_rate) {
        int out_h = grad_output.size();
        int out_w = grad_output[0].size();
        int in_h = this->input.size();
        int in_w = this->input[0].size();

        // 1. Temporary storage for gradients (so we don't change weights mid-loop)
        vector<vector<double>> g_weights(k_size, vector<double>(k_size, 0.0));
        vector<vector<double>> grad_input(in_h, vector<double>(in_w, 0.0));
        double g_bias = 0.0;

        // 2. The Accumulation Loop
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                g_bias += grad_output[i][j];

                for (int ki = 0; ki < k_size; ki++) {
                    for (int kj = 0; kj < k_size; kj++) {
                        // Accumulate gradient for filter
                        g_weights[ki][kj] += this->input[i + ki][j + kj] * grad_output[i][j];
                        
                        // Accumulate gradient for the previous layer
                        // (Uses the CURRENT filter values)
                        grad_input[i + ki][j + kj] += this->filter[ki][kj] * grad_output[i][j];
                    }
                }
            }
        }

        // 3. Update the weights ONLY AFTER the loops are completely finished
        for (int ki = 0; ki < k_size; ki++) {
            for (int kj = 0; kj < k_size; kj++) {
                this->filter[ki][kj] -= learning_rate * g_weights[ki][kj];
            }
        }
        this->bias -= learning_rate * g_bias;

        // 4. Return the error to the previous layer
        return grad_input;
    }
};

// Done --- Fully Connected Layer (1D) ---
class FCLayer {
public:
    vector<vector<double>> weights;
    vector<double> bias;
    vector<double> input;
    int in_size, out_size;

    FCLayer(int in_size, int out_size) {
        // We use this-> to distinguish the class member from the function parameter
        this->in_size = in_size;
        this->out_size = out_size;

        // Initialize weights and biases
        this->weights.assign(this->out_size, vector<double>(this->in_size));
        this->bias.assign(this->out_size, 0.0);

        for (int i = 0; i < this->out_size; i++) {
            // Accessing bias member
            this->bias[i] = ((rand() % 2000) / 10000.0) - 0.1;
            
            for (int j = 0; j < this->in_size; j++) {
                // Accessing weights member
                this->weights[i][j] = ((rand() % 2000) / 10000.0) - 0.1;
            }
        }
    }

    vector<double> forward(const vector<double>& x) {
        input = x; // Cache input for the backward pass
        vector<double> output(out_size, 0.0);

        for (int i = 0; i < out_size; i++) {
            double sum = bias[i];
            for (int j = 0; j < in_size; j++) {
                sum += x[j] * weights[i][j];
            }
            output[i] = sum;
        }
        return output;
    }

    vector<double> backward(const vector<double>& grad_output, double learning_rate) {
        vector<double> grad_input(in_size, 0.0);

        // 1. Calculate grad_input (error to pass back to the previous layer)
        // This must be done BEFORE we update the weights
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                grad_input[j] += weights[i][j] * grad_output[i];
            }
        }

        // 2. Perform Gradient Descent: Update Weights and Bias
        // Formula: w = w - lr * (input * grad_output)
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                double weight_gradient = input[j] * grad_output[i];
                weights[i][j] -= learning_rate * weight_gradient;
            }
            // Bias gradient is just the grad_output itself
            bias[i] -= learning_rate * grad_output[i];
        }

        return grad_input;
    }
};

// Done Flattening Layer (The Bridge) ---
class FlattenLayer {
    public:
        int in_h, in_w;

        vector<double> forward(const vector<vector<double>>& x) {
            in_h = x.size();
            in_w = x[0].size();
            
            vector<double> flat;
            for(int i = 0; i < in_h; i++) {
                for(int j = 0; j < in_w; j++) {
                    flat.push_back(x[i][j]);
                }
            }
            return flat;
        }

        vector<vector<double>> backward(const vector<double>& grad_output) {
            vector<vector<double>> grad_2d(in_h, vector<double>(in_w));
            int k = 0;
            for (int i = 0; i < in_h; i++) {
                for (int j = 0; j < in_w; j++) {
                    grad_2d[i][j] = grad_output[k];
                    k++;
                }
            }
            return grad_2d;
        }
};

// Done MaxPool Layer (2D) ---
class MaxPoolLayer {
    public:
        vector<vector<double>> input;
        int stride = 2;

        vector<vector<double>> forward(const vector<vector<double>>& x) {
            input = x; // Cache the full input for backprop
            int in_h = x.size();
            int in_w = x[0].size();
            int out_h = in_h / stride;
            int out_w = in_w / stride;

            vector<vector<double>> output(out_h, vector<double>(out_w, 0.0));

            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    double max_val = -1e9; // Very small number
                    for (int m = 0; m < stride; m++) {
                        for (int n = 0; n < stride; n++) {
                            max_val = max(max_val, x[i * stride + m][j * stride + n]);
                        }
                    }
                    output[i][j] = max_val;
                }
            }
            return output;
        }

        vector<vector<double>> backward(const vector<vector<double>>& grad_output) {
            int in_h = input.size();
            int in_w = input[0].size();
            int out_h = grad_output.size();
            int out_w = grad_output[0].size();

            // Initialize grad_input with zeros
            vector<vector<double>> grad_input(in_h, vector<double>(in_w, 0.0));

            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    // Find the location of the max value in the original input
                    int max_i = i * stride;
                    int max_j = j * stride;
                    double max_val = input[max_i][max_j];

                    for (int m = 0; m < stride; m++) {
                        for (int n = 0; n < stride; n++) {
                            if (input[i * stride + m][j * stride + n] > max_val) {
                                max_val = input[i * stride + m][j * stride + n];
                                max_i = i * stride + m;
                                max_j = j * stride + n;
                            }
                        }
                    }
                    // Pass the gradient only to that specific "winning" pixel
                    grad_input[max_i][max_j] = grad_output[i][j];
                }
            }
            return grad_input;
        }
};


// Done ReLU Layer (2D) --- (here in backprop we pass gradient only where input > 0, because ReLU derivative is 1 there, else 0)
class ReLULayer {
    public:
        vector<vector<double>> input;

        vector<vector<double>> forward(const vector<vector<double>>& x) {
            input = x; // Cache for backward pass
            int h = x.size();
            int w = x[0].size();
            vector<vector<double>> output(h, vector<double>(w));

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    // ReLU formula: max(0, x)
                    output[i][j] = (x[i][j] > 0) ? x[i][j] : 0;
                }
            }
            return output;
        }

        vector<vector<double>> backward(const vector<vector<double>>& grad_output) {
            int h = input.size();
            int w = input[0].size();
            vector<vector<double>> grad_input(h, vector<double>(w));

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    // If input was > 0, pass the gradient. Otherwise, it's 0.
                    if (input[i][j] > 0) {
                        grad_input[i][j] = grad_output[i][j];
                    } else {
                        grad_input[i][j] = 0;
                    }
                }
            }
            return grad_input;
        }
};


// Done Sigmoid Layer (1D) ---
//backpropagation formula: (we can simply pass the same gradient back, but we need to multiply by the derivative of the sigmoid function)
//$$\frac{d}{dx}\sigma(x) = \sigma(x) \cdot (1 - \sigma(x))$$
//grad_input[i] = grad_output[i] * sigmoid_derivative;
class SigmoidLayer {
    public:
        vector<double> output; 

        vector<double> forward(const vector<double>& x) {
            int n = x.size();
            output.assign(n, 0.0);
            
            for (int i = 0; i < n; i++) {
                // Formula: 1 / (1 + e^-x)
                output[i] = 1.0 / (1.0 + exp(-x[i]));
            }
            return output;
        }

        vector<double> backward(const vector<double>& grad_output) {
            int n = output.size();
            vector<double> grad_input(n);

            for (int i = 0; i < n; i++) {
                // Math: error * derivative
                // Derivative of sigmoid is: sigmoid(x) * (1 - sigmoid(x))
                double sigmoid_derivative = output[i] * (1.0 - output[i]);
                grad_input[i] = grad_output[i] * sigmoid_derivative;
            }
            return grad_input;
        }
};

// Done Loss Function ---
class Loss {
    public:
        // 1. Calculate the total Error (Scalar value for us to see progress)
        double calculate(vector<double> pred, vector<double> target) {
            double sum = 0.0;
            int n = pred.size();
            for (int i = 0; i < n; i++) {
                double diff = pred[i] - target[i];
                sum += diff * diff;
            }
            return sum / n; // Average squared error
        }

        // 2. The "Starting Error" for Backprop
        vector<double> backward(vector<double> pred, vector<double> target) {
            int n = pred.size();
            vector<double> grad_output(n);
            
            for (int i = 0; i < n; i++) {
                // Derivative of (pred - target)^2 is 2 * (pred - target)
                // We usually ignore the '2' or fold it into the learning rate
                grad_output[i] = (pred[i] - target[i]);
            }
            return grad_output;
        }
};




// Only Dummy : The Manager Model ---
class CNNModel {
public:
    ConvLayer conv1;
    ReLULayer relu1;
    MaxPoolLayer pool1;
    FlattenLayer flatten;
    FCLayer fc1;
    SigmoidLayer sig1;
    Loss loss_fn;

    // Constructor params based on dataset (28 or 32)
    CNNModel() : 
        conv1(1, 8, 3), 
        fc1(1352, 10) // Example: 13x13x8 = 1352
    {}

    vector<double> forward(vector<vector<double>> x) {
        auto out_conv = conv1.forward(x);
        auto out_relu = relu1.forward(out_conv);
        auto out_pool = pool1.forward(out_relu);
        auto out_flat = flatten.forward(out_pool);
        auto out_fc   = fc1.forward(out_flat);
        auto final    = sig1.forward(out_fc);
        return final;
    }

    void backward(vector<double> initial_grad) {
        auto g_sig   = sig1.backward(initial_grad);
        auto g_fc    = fc1.backward(g_sig);
        auto g_flat  = flatten.backward(g_fc);
        auto g_pool  = pool1.backward(g_flat);
        auto g_relu  = relu1.backward(g_pool);
        conv1.backward(g_relu);
    }
};

int main() {
    CNNModel model;
    
    // Dataset 1: 2D representation
    vector<vector<double>> image(28, vector<double>(28, 0.1));
    vector<double> target(10, 0.0); target[3] = 1.0; // One-hot example

    // --- Training Step ---
    vector<double> prediction = model.forward(image);
    vector<double> error_grad = model.loss_fn.backward(prediction, target);
    model.backward(error_grad);

    cout << "Unified 2D/1D CNN Model Skeleton is ready." << endl;
    return 0;
}