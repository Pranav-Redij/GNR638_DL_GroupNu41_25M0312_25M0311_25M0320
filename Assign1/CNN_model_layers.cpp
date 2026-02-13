#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>

using namespace std;

class ReLU1DLayer {
public:
    vector<double> input;

    vector<double> forward(const vector<double>& x) {
        input = x;
        vector<double> out(x.size());

        for (int i = 0; i < x.size(); i++)
            out[i] = (x[i] > 0) ? x[i] : 0;

        return out;
    }

    vector<double> backward(const vector<double>& grad_output) {
        vector<double> grad_input(input.size());

        for (int i = 0; i < input.size(); i++) {
            grad_input[i] = (input[i] > 0) ? grad_output[i] : 0;
        }

        return grad_input;
    }
};

// --- Multi-Filter Multi-Channel Convolution Layer ---
class ConvLayer {
public:
    int k_size;
    int in_channels;
    int num_filters;

    // [F][C][k][k]
    vector<vector<vector<vector<double>>>> filters;

    // [F]
    vector<double> biases;

    // [C][H][W]
    vector<vector<vector<double>>> input;

    ConvLayer(int k_size, int in_channels, int num_filters) {
        this->k_size = k_size;
        this->in_channels = in_channels;
        this->num_filters = num_filters;

        filters.resize(num_filters,
            vector<vector<vector<double>>>(
                in_channels,
                vector<vector<double>>(
                    k_size,
                    vector<double>(k_size)
                )
            )
        );

        biases.resize(num_filters);

        for (int f = 0; f < num_filters; f++) {

            biases[f] = ((rand() % 2000) / 10000.0) - 0.1;

            for (int c = 0; c < in_channels; c++) {
                for (int i = 0; i < k_size; i++) {
                    for (int j = 0; j < k_size; j++) {
                        filters[f][c][i][j] =
                            ((rand() % 2000) / 10000.0) - 0.1;
                    }
                }
            }
        }
    }

    long getParams() {
        return (long)(num_filters * in_channels * k_size * k_size)
               + num_filters;
    }

    long getMACs(int in_h, int in_w) {
        int out_h = in_h - k_size + 1;
        int out_w = in_w - k_size + 1;

        return (long)num_filters *
               out_h * out_w *
               k_size * k_size * in_channels;
    }

    // ---------- FORWARD ----------
    vector<vector<vector<double>>> forward(
        const vector<vector<vector<double>>>& x)
    {
        input = x;

        int in_h = x[0].size();
        int in_w = x[0][0].size();

        int out_h = in_h - k_size + 1;
        int out_w = in_w - k_size + 1;

        // [F][H][W]
        vector<vector<vector<double>>> output(
            num_filters,
            vector<vector<double>>(out_h,
                vector<double>(out_w, 0.0)));

        for (int f = 0; f < num_filters; f++) {

            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {

                    double sum = biases[f];

                    for (int c = 0; c < in_channels; c++) {
                        for (int ki = 0; ki < k_size; ki++) {
                            for (int kj = 0; kj < k_size; kj++) {

                                sum += x[c][i + ki][j + kj] *
                                       filters[f][c][ki][kj];
                            }
                        }
                    }

                    output[f][i][j] = sum;
                }
            }
        }

        return output;
    }

    // ---------- BACKWARD ----------
    vector<vector<vector<double>>> backward(
        const vector<vector<vector<double>>>& grad_output,
        double learning_rate)
    {
        int out_h = grad_output[0].size();
        int out_w = grad_output[0][0].size();

        int in_h = input[0].size();
        int in_w = input[0][0].size();

        // [C][H][W]
        vector<vector<vector<double>>> grad_input(
            in_channels,
            vector<vector<double>>(in_h,
                vector<double>(in_w, 0.0)));

        // [F][C][k][k]
        vector<vector<vector<vector<double>>>> grad_weights(
            num_filters,
            vector<vector<vector<double>>>(
                in_channels,
                vector<vector<double>>(
                    k_size,
                    vector<double>(k_size, 0.0)
                )
            )
        );

        vector<double> grad_bias(num_filters, 0.0);

        for (int f = 0; f < num_filters; f++) {

            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {

                    double go = grad_output[f][i][j];
                    grad_bias[f] += go;

                    for (int c = 0; c < in_channels; c++) {
                        for (int ki = 0; ki < k_size; ki++) {
                            for (int kj = 0; kj < k_size; kj++) {

                                grad_weights[f][c][ki][kj] +=
                                    input[c][i + ki][j + kj] * go;

                                grad_input[c][i + ki][j + kj] +=
                                    filters[f][c][ki][kj] * go;
                            }
                        }
                    }
                }
            }
        }

        // Update weights
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < in_channels; c++) {
                for (int ki = 0; ki < k_size; ki++) {
                    for (int kj = 0; kj < k_size; kj++) {
                        filters[f][c][ki][kj] -=
                            learning_rate * grad_weights[f][c][ki][kj];
                    }
                }
            }

            biases[f] -= learning_rate * grad_bias[f];
        }

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
        this->in_size = in_size;
        this->out_size = out_size;

        weights.assign(out_size, vector<double>(in_size));
        bias.assign(out_size, 0.0);

        for (int i = 0; i < out_size; i++) {
            bias[i] = ((rand() % 2000) / 10000.0) - 0.1;

            for (int j = 0; j < in_size; j++) {
                weights[i][j] =
                    ((rand() % 2000) / 10000.0) - 0.1;
            }
        }
    }

    long getParams() {
        return (long)in_size * out_size + out_size;
    }

    long getMACs() {
        return (long)in_size * out_size;
    }

    vector<double> forward(const vector<double>& x) {
        input = x;

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

    vector<double> backward(
        const vector<double>& grad_output,
        double learning_rate)
    {
        vector<double> grad_input(in_size, 0.0);

        // Compute gradient wrt input (before updating weights)
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                grad_input[j] +=
                    weights[i][j] * grad_output[i];
            }
        }

        // Update weights & bias
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                weights[i][j] -=
                    learning_rate *
                    input[j] * grad_output[i];
            }

            bias[i] -= learning_rate * grad_output[i];
        }

        return grad_input;
    }
};


// Done Flattening Layer (The Bridge) ---
class FlattenLayer {
public:
    int in_c, in_h, in_w;

    // Forward: 3D → 1D
    vector<double> forward(
        const vector<vector<vector<double>>>& x)
    {
        in_c = x.size();
        in_h = x[0].size();
        in_w = x[0][0].size();

        vector<double> flat;
        flat.reserve(in_c * in_h * in_w);

        for (int c = 0; c < in_c; c++) {
            for (int i = 0; i < in_h; i++) {
                for (int j = 0; j < in_w; j++) {
                    flat.push_back(x[c][i][j]);
                }
            }
        }

        return flat;
    }

    // Backward: 1D → 3D
    vector<vector<vector<double>>> backward(
        const vector<double>& grad_output)
    {
        vector<vector<vector<double>>> grad_input(
            in_c,
            vector<vector<double>>(in_h,
            vector<double>(in_w)));

        int idx = 0;

        for (int c = 0; c < in_c; c++) {
            for (int i = 0; i < in_h; i++) {
                for (int j = 0; j < in_w; j++) {
                    grad_input[c][i][j] =
                        grad_output[idx++];
                }
            }
        }

        return grad_input;
    }
};



// Done MaxPool Layer (2D) ---
class MaxPoolLayer {
public:
    vector<vector<vector<double>>> input;
    int stride = 2;

    // -------- FORWARD --------
    vector<vector<vector<double>>> forward(
        const vector<vector<vector<double>>>& x)
    {
        input = x;

        int C = x.size();
        int H = x[0].size();
        int W = x[0][0].size();

        int out_h = H / stride;
        int out_w = W / stride;

        vector<vector<vector<double>>> output(
            C,
            vector<vector<double>>(out_h,
            vector<double>(out_w)));

        for (int c = 0; c < C; c++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {

                    double max_val = -1e15;

                    for (int m = 0; m < stride; m++) {
                        for (int n = 0; n < stride; n++) {

                            max_val = max(
                                max_val,
                                x[c][i*stride + m][j*stride + n]
                            );
                        }
                    }

                    output[c][i][j] = max_val;
                }
            }
        }

        return output;
    }

    // -------- BACKWARD --------
    vector<vector<vector<double>>> backward(
        const vector<vector<vector<double>>>& grad_output)
    {
        int C = input.size();
        int H = input[0].size();
        int W = input[0][0].size();

        int out_h = grad_output[0].size();
        int out_w = grad_output[0][0].size();

        vector<vector<vector<double>>> grad_input(
            C,
            vector<vector<double>>(H,
            vector<double>(W, 0.0)));

        for (int c = 0; c < C; c++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {

                    int base_i = i * stride;
                    int base_j = j * stride;

                    int max_i = base_i;
                    int max_j = base_j;
                    double max_val = input[c][base_i][base_j];

                    for (int m = 0; m < stride; m++) {
                        for (int n = 0; n < stride; n++) {

                            int cur_i = base_i + m;
                            int cur_j = base_j + n;

                            if (input[c][cur_i][cur_j] > max_val) {
                                max_val = input[c][cur_i][cur_j];
                                max_i = cur_i;
                                max_j = cur_j;
                            }
                        }
                    }

                    grad_input[c][max_i][max_j] =
                        grad_output[c][i][j];
                }
            }
        }

        return grad_input;
    }
};




// Done ReLU Layer (2D) --- (here in backprop we pass gradient only where input > 0, because ReLU derivative is 1 there, else 0)
class ReLULayer {
public:
    vector<vector<vector<double>>> input;

    // -------- FORWARD --------
    vector<vector<vector<double>>> forward(
        const vector<vector<vector<double>>>& x)
    {
        input = x;

        int C = x.size();
        int H = x[0].size();
        int W = x[0][0].size();

        vector<vector<vector<double>>> output(
            C,
            vector<vector<double>>(H,
            vector<double>(W)));

        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {

                    output[c][i][j] =
                        (x[c][i][j] > 0.0) ?
                        x[c][i][j] : 0.0;
                }
            }
        }

        return output;
    }

    // -------- BACKWARD --------
    vector<vector<vector<double>>> backward(
        const vector<vector<vector<double>>>& grad_output)
    {
        int C = input.size();
        int H = input[0].size();
        int W = input[0][0].size();

        vector<vector<vector<double>>> grad_input(
            C,
            vector<vector<double>>(H,
            vector<double>(W)));

        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {

                    grad_input[c][i][j] =
                        (input[c][i][j] > 0.0)
                        ? grad_output[c][i][j]
                        : 0.0;
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

/*class SigmoidLayer {
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
};*/

// Done Loss Function ---
/*
This combines:
Softmax (inside)
CrossEntropy
Correct gradient
*/
class Loss {
public:

    // -------- SOFTMAX --------
    vector<double> softmax(const vector<double>& x) {

        int n = x.size();
        vector<double> probs(n);

        double max_val = *max_element(x.begin(), x.end());

        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            probs[i] = exp(x[i] - max_val);
            sum += probs[i];
        }

        for (int i = 0; i < n; i++) {
            probs[i] /= sum;
        }

        return probs;
    }

    // -------- CROSS ENTROPY --------
    double calculate(const vector<double>& logits,
                     const vector<double>& target)
    {
        vector<double> probs = softmax(logits);

        double loss = 0.0;
        const double eps = 1e-9;

        for (int i = 0; i < probs.size(); i++) {
            if (target[i] == 1.0) {
                loss -= log(probs[i] + eps);
            }
        }

        return loss;
    }

    // -------- GRADIENT --------
    vector<double> backward(const vector<double>& logits,
                            const vector<double>& target)
    {
        vector<double> probs = softmax(logits);

        int n = probs.size();
        vector<double> grad(n);

        for (int i = 0; i < n; i++) {
            grad[i] = probs[i] - target[i];
        }

        return grad;
    }
};



int main() {

    return 0;
}

