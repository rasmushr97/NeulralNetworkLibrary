package neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    private List<Double> deltaWs = new ArrayList<>();
    private List<Double> weights;
    private List<Double> input;
    private ActivationFunction activationFunction;
    private double bias;
    private double output = 0;
    private double error;

    Neuron(int inputSize, ActivationFunction activationFunction) {
        Random random = new Random();
        this.activationFunction = activationFunction;
        this.bias = random.nextDouble() * 2 - 1;


        weights = new ArrayList<>();
        for (int i = 0; i < inputSize; i++) {
            double weight = random.nextDouble() * 2 - 1;
            weights.add(weight);
        }
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getError() {
        return error;
    }

    void setInput(List<Double> input) {
        this.input = input;
    }

    List<Double> getWeights() {
        return weights;
    }

    double getBias() {
        return bias;
    }

    void performBackprop() {
        deltaWs = new ArrayList<>();
        for (double in : input) {
            double deltaW = in * activationFunction.derivative(output) * error;
            deltaW *= NeuralNetwork.learningRate;
            deltaWs.add(deltaW);
        }
        weights = vectorAddition(weights, deltaWs);
        bias += activationFunction.derivative(output) * error * NeuralNetwork.learningRate;
    }

    double getWeightSum() {
        return weights
                .stream()
                .mapToDouble(x -> x)
                .sum();
    }

    double getOutput() {
        return output;
    }

    void calc() {
        output = dotProduct(input, weights) + bias;
        output = activationFunction.calc(output);
    }

    @Override
    public String toString() {
        return "neuralnet.Neuron{" +
                "weights=" + weights +
                ", delta Ws=" + deltaWs +
                ", bias=" + bias +
                '}';
    }


    private double dotProduct(List<Double> vectorA, List<Double> vectorB) {
        if (vectorA.size() != vectorB.size()) {
            throw new RuntimeException("Incompatible lists");
        }

        double res = 0;
        for (int i = 0; i < vectorA.size(); i++) {
            res += vectorA.get(i) * vectorB.get(i);
        }

        return res;
    }

    private List<Double> vectorAddition(List<Double> a, List<Double> b) {
        if (a.size() != b.size()) {
            throw new RuntimeException("Vector addition not possible with vectors of different sizes");
        }
        List<Double> res = new ArrayList<>();

        for (int i = 0; i < a.size(); i++) {
            double added = a.get(i) + b.get(i);
            res.add(added);
        }

        return res;
    }

}











