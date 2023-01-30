#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>

using std::cout, std::cin, std::vector, std::string, std::fstream, std::stringstream, std::ios;

//for test
//const vector<int> LAYERS{3, 2, 2};

const vector<int> LAYERS{100, 100, 10};
const double LEARNING_RATE = .3;
const int EPOCHS = 10000;
// const int BATCH_SIZE = 32;

void print(const vector<vector<vector<double>>> &vec) {
    for (const auto &i: vec) {
        for (const auto &j: i) {
            for (const auto &k: j)
                cout << k << ", ";

            cout << "\n";
        }
        cout << "\n";
    }
}

void print(const vector<vector<double>> &vec) {
    for (const auto & i : vec) {
        for (double j : i) {
            cout << j << ", ";
        }
        cout << "\n";
    }
}

void print(const vector<double> &vec) {
    for (double i : vec) {
        cout << i << ", ";
    }
    cout << "\n";
}

void print(const vector<int> &vec) {
    for (int i : vec) {
        cout << i << ", ";
    }
    cout << "\n";
}

void initializeRandomNumber(vector<double> &vec) {
    for (auto &el: vec)
        el = ((double) rand() / RAND_MAX) * 2 - 1;
}

vector<vector<string>> getData(const string &fileName) {
    vector<vector<string>> content;
    vector<string> row;
    string line, word;

    fstream file(fileName, ios::in);
    if (file.is_open()) {
        getline(file, line);
        while (getline(file, line)) {
            row.clear();
            stringstream str(line);

            while (getline(str, word, ','))
                row.push_back(word);

            content.push_back(row);
        }
    } else
        cout << "Could not open the file\n";

    file.close();

    return content;
}

void init(vector<double> &input, vector<vector<double>> &hiddenLayers, vector<double> &output,
          vector<vector<vector<double>>> &weights) {
    // initialize the hidden layers
    for (int i = 0; i < LAYERS.size() - 1; i++) {
        vector<double> hiddenLayer(LAYERS[i]);
        hiddenLayers[i] = hiddenLayer;
    }

    // initialize the weights
    for (int i = 0; i < LAYERS.size(); i++) {
        if (i == 0) {
            vector<vector<double>> w;
            for (int j = 0; j < input.size() + 1; j++) {
                vector<double> row(LAYERS[0]);
                initializeRandomNumber(row);
                w.push_back(row);
            }
            weights.push_back(w);
        } else {
            vector<vector<double>> w;
            for (int j = 0; j < LAYERS[i - 1] + 1; j++) {
                vector<double> row(LAYERS[i]);
                initializeRandomNumber(row);
                w.push_back(row);
            }
            weights.push_back(w);
        }
    }
}

vector<bool> oneHotEncoding(int y) {
    int outputCount = LAYERS[LAYERS.size() - 1];
    vector<bool> encoding(outputCount, false);
    encoding[y] = true;
    return encoding;
}

double sigmoid(double z) {
    return 1.0 / (1 + exp(-z));
}

double relue(double z) {
    return z >= 0 ? z : 0;
}

void feetForwardWithSoftmax(const vector<double> &input, vector<vector<double>> &hiddenLayers,
                            vector<double> &output, const vector<vector<vector<double>>> &weights) {
    for (int hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayers.size(); hiddenLayerIndex++) {
        const vector<vector<double>> &weightsOfHiddenLayer = weights[hiddenLayerIndex];
        for (int neuron = 0; neuron < hiddenLayers[hiddenLayerIndex].size(); neuron++) {
            double sum = 0;
            sum += weightsOfHiddenLayer[0][neuron];
            if (hiddenLayerIndex == 0) {
                for (int i = 0; i < input.size(); i++)
                    sum += input[i] * weightsOfHiddenLayer[i + 1][neuron];
            } else {
                for (int i = 0; i < hiddenLayers[hiddenLayerIndex - 1].size(); i++)
                    sum += hiddenLayers[hiddenLayerIndex - 1][i] * weightsOfHiddenLayer[i + 1][neuron];
            }
            hiddenLayers[hiddenLayerIndex][neuron] = sigmoid(sum);
            // cout << sum << " " << hiddenLayers[hiddenLayerIndex][neuron] << " ---\n";
        }
    }

    auto &weightsForOutputLayer = weights[weights.size() - 1];
    auto &lastHiddenLayer = hiddenLayers[hiddenLayers.size() - 1];
    double totalExpSum = 0;
    for (int outputNeuron = 0; outputNeuron < output.size(); outputNeuron++) {
        double sum = 0;
        sum += weightsForOutputLayer[0][outputNeuron];
        for (int i = 0; i < lastHiddenLayer.size(); i++)
            sum += lastHiddenLayer[i] * weightsForOutputLayer[i + 1][outputNeuron];

        output[outputNeuron] = exp(sum);
        totalExpSum += output[outputNeuron];
    }
    for (auto &o: output)
        o /= totalExpSum;
}

void backPropagationWithSoftmax(const vector<double> &input, const vector<vector<double>> &hiddenLayers,
                                const vector<double> &output, vector<vector<vector<double>>> &weights,
                                const vector<bool> &oneHotEncodingY) {
    vector<vector<double>> dcOverdz;
    for (int i : LAYERS) {
        vector<double> dciOverdzi(i);
        dcOverdz.push_back(dciOverdzi);
    }

    // find the derivative of the output layer
    for (int outputIndex = 0; outputIndex < output.size(); outputIndex++) {
        dcOverdz[dcOverdz.size() - 1][outputIndex] = output[outputIndex] - oneHotEncodingY[outputIndex];
    }

    // find the derivative of the hidden layers
    for (int hiddenLayerIndex = hiddenLayers.size() - 1; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
        for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayers[hiddenLayerIndex].size(); hiddenNeuron++) {
            // find the sum
            double sum = 0;
            if (hiddenLayerIndex == hiddenLayers.size() - 1) {
                for (int outputIndex = 0; outputIndex < output.size(); outputIndex++) {
                    sum += weights[hiddenLayerIndex + 1][hiddenNeuron + 1][outputIndex] *
                           dcOverdz[hiddenLayerIndex + 1][outputIndex];
                }
            } else {
                for (int outputIndex = 0; outputIndex < hiddenLayers[hiddenLayerIndex + 1].size(); outputIndex++) {
                    sum += weights[hiddenLayerIndex + 1][hiddenNeuron + 1][outputIndex] *
                           dcOverdz[hiddenLayerIndex + 1][outputIndex];
                }
            }
            dcOverdz[hiddenLayerIndex][hiddenNeuron] =
                    hiddenLayers[hiddenLayerIndex][hiddenNeuron] * (1 - hiddenLayers[hiddenLayerIndex][hiddenNeuron]) *
                    sum;
        }
    }

    // update the weights of the output layer
    for (int outputIndex = 0; outputIndex < output.size(); outputIndex++) {
        // Wio = Wio - learningRate * dcOverdzo * input
        weights[weights.size() - 1][0][outputIndex] -=
                LEARNING_RATE * dcOverdz[dcOverdz.size() - 1][outputIndex];

        for (int i = 0; i < hiddenLayers[hiddenLayers.size() - 1].size(); i++)
            weights[weights.size() - 1][i + 1][outputIndex] -=
                    LEARNING_RATE * dcOverdz[dcOverdz.size() - 1][outputIndex] *
                    hiddenLayers[hiddenLayers.size() - 1][i];
    }

    // update the weights of the hidden layers except first one
    for (int hiddenLayerIndex = hiddenLayers.size() - 1; hiddenLayerIndex > 0; hiddenLayerIndex--) {
        for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayers[hiddenLayerIndex].size(); hiddenNeuron++) {
            weights[hiddenLayerIndex][0][hiddenNeuron] -=
                    LEARNING_RATE * dcOverdz[hiddenLayerIndex][hiddenNeuron];

            for (int i = 0; i < hiddenLayers[hiddenLayerIndex - 1].size(); i++) {
                weights[hiddenLayerIndex][i + 1][hiddenNeuron] -=
                        LEARNING_RATE * dcOverdz[hiddenLayerIndex][hiddenNeuron] *
                        hiddenLayers[hiddenLayerIndex - 1][i];
            }
        }
    }

    // update the weights of the first hidden layer
    for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayers[0].size(); hiddenNeuron++) {
        weights[0][0][hiddenNeuron] -=
                LEARNING_RATE * dcOverdz[0][hiddenNeuron];

        for (int i = 0; i < input.size(); i++) {
            weights[0][i + 1][hiddenNeuron] -=
                    LEARNING_RATE * dcOverdz[0][hiddenNeuron] * input[i];
        }
    }
}

int main() {
    auto data = getData("./digit-recognizer/train.csv");

// for test
// XOR
//    vector<vector<string>> data = {
//            {"0", "0", "0"},
//            {"1", "0", "1"},
//            {"1", "1", "0"},
//            {"0", "1", "1"}
//    };

    const int inputSize = data[0].size() - 1;

    vector<double> input(inputSize);
    vector<vector<double>> hiddenLayers(LAYERS.size() - 1);
    vector<vector<vector<double>>> weights;
    vector<double> output(LAYERS[LAYERS.size() - 1]);

    init(input, hiddenLayers, output, weights);

    // for test
//    weights[0][0][0] = .1;
//    weights[0][0][1] = .2;
//    weights[0][0][2] = .3;
//    weights[0][1][0] = .4;
//    weights[0][1][1] = .5;
//    weights[0][1][2] = .6;
//    weights[0][2][0] = .7;
//    weights[0][2][1] = .8;
//    weights[0][2][2] = .9;
//
//    weights[1][0][0] = .15;
//    weights[1][0][1] = .85;
//    weights[1][1][0] = .64;
//    weights[1][1][1] = .88;
//    weights[1][2][0] = .28;
//    weights[1][2][1] = .39;
//    weights[1][3][0] = .45;
//    weights[1][3][1] = .56;
//
//    weights[2][0][0] = .77;
//    weights[2][0][1] = .88;
//    weights[2][1][0] = .11;
//    weights[2][1][1] = .39;
//    weights[2][2][0] = .67;
//    weights[2][2][1] = .87;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double error = 0;
        for (const auto& patterData : data) {
            int y = stoi(patterData[0]);
            vector<bool> oneHotEncodingY = oneHotEncoding(y);

            for (int i = 0; i < input.size(); i++)
                input[i] = std::stod(patterData[i + 1]) / 255;

            feetForwardWithSoftmax(input, hiddenLayers, output, weights);
            backPropagationWithSoftmax(input, hiddenLayers, output, weights, oneHotEncodingY);

            error += -log(output[y]);
        }
        cout << error << " epoch >>> " << epoch << "\n";
    }
    return 0;
}
