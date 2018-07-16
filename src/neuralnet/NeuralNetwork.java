package neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    private List<HiddenLayer> neuralLayers;
    private List<Integer> layerSizes;
    public static double learningRate = 0.05;
    private List<List<Double>> trainingInputs;
    private List<List<Double>> trainingTargets;
    private List<List<Double>> testingInputs;
    private List<List<Double>> testingTargets;
    private boolean softmaxIsEnabled = false;
    private ActivationFunction activationFunction = new Sigmoid();

    public NeuralNetwork(Integer... layers) {
        layerSizes = Arrays.asList(layers);
        neuralLayers = new ArrayList<>();

        validateInput();

        for (int i = 1; i < layerSizes.size(); i++) {
            HiddenLayer layer = new HiddenLayer(activationFunction, layerSizes.get(i - 1), layerSizes.get(i));
            neuralLayers.add(layer);
        }
    }

    public void setTrainingData(List<List<Double>> inputs, List<List<Double>> targets) {
        this.trainingInputs = inputs;
        this.trainingTargets = targets;
    }

    public void setTestingData(List<List<Double>> inputs, List<List<Double>> targets) {
        this.testingInputs = inputs;
        this.testingTargets = targets;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void setSoftmaxIsEnabled(boolean softmaxEnabled) {
        this.softmaxIsEnabled = softmaxEnabled;
    }

    public void split(List<List<Double>> inputs, List<List<Double>> targets) {
        int length = inputs.size();
        int trainingDataSize = (int) (0.8 * length);

        trainingInputs = inputs.subList(0, trainingDataSize);
        trainingTargets = targets.subList(0, trainingDataSize);

        testingInputs = inputs.subList(trainingDataSize, length);
        testingTargets = targets.subList(trainingDataSize, length);
    }

    public void split(List<List<Double>> inputs, List<List<Double>> targets, float splitRatio) {
        int length = inputs.size();
        int trainingDataSize = (int) (splitRatio * length);

        trainingInputs = inputs.subList(0, trainingDataSize);
        trainingTargets = targets.subList(0, trainingDataSize);

        testingInputs = inputs.subList(trainingDataSize, length);
        testingTargets = targets.subList(trainingDataSize, length);
    }

    public List<Double> makePrediction(List<Double> input) {
        List<Double> prediction = input;
        for (HiddenLayer neuralLayer : neuralLayers) {
            neuralLayer.setInput(prediction);
            neuralLayer.calc();
            prediction = neuralLayer.getOutput();
        }

        if(softmaxIsEnabled){
            prediction = softmax(prediction);
        }

        return prediction;
    }

    public void train() {
        if (trainingInputs == null || testingTargets == null) {
            throw new RuntimeException("Training data not set");
        } else if (trainingInputs.size() != trainingTargets.size()) {
            String message = "The amount of inputs has to be the same as the amount of targets";
            throw new RuntimeException(message);
        }

        int prevPercentage = 0;
        for (int i = 0; i < trainingInputs.size(); i++) {
            int percentage = (int) ((float) i / (float) trainingInputs.size() * 100);
            if(percentage != prevPercentage){
                System.out.println(percentage + "%");
            }
            prevPercentage = percentage;

            List<Double> output = this.makePrediction(trainingInputs.get(i));
            List<Double> target = trainingTargets.get(i);

            if (target.size() != output.size()) {
                throw new RuntimeException("The size of the target data does not match that of the output");
            }

            setErrors(output, target);
            for (HiddenLayer neuralLayer : neuralLayers) {
                neuralLayer.performBackprop();
            }

        }
    }

    public double test() {
        int correctPredictions = 0;

        for (int i = 0; i < testingInputs.size(); i++) {
            List<Double> input = testingInputs.get(i);
            List<Double> output = this.makePrediction(input);
            List<Double> target = testingTargets.get(i);

            int targetMaxIndex = indexOfMax(target);
            int outputMaxIndex = indexOfMax(output);

            if (targetMaxIndex == outputMaxIndex) {
                correctPredictions++;
            }
        }

        return (double) correctPredictions / (double) testingInputs.size();
    }


    private void setErrors(List<Double> output, List<Double> target) {
        List<Double> prevErrors;
        HiddenLayer prevLayer = neuralLayers.get(neuralLayers.size() - 1);

        // Find the errors of the last layers and put them in a list
        List<Double> lastLayerErrors = new ArrayList<>();
        for (int i = 0; i < output.size(); i++) {
            double error = target.get(i) - output.get(i);
            lastLayerErrors.add(error);
        }
        prevLayer.setErrors(lastLayerErrors);

        for (int i = neuralLayers.size() - 2; i >= 0; i--) {
            HiddenLayer neuralLayer = neuralLayers.get(i);
            List<List<Double>> errorsToBeSummed = new ArrayList<>();

            List<Neuron> neurons = prevLayer.getNeurons();
            for (int j = 0; j < neurons.size(); j++) {
                Neuron neuron = neurons.get(j);
                List<Double> errors = new ArrayList<>();

                for (double weight : neuron.getWeights()) {
                    double error = weight * neuron.getError();
                    errors.add(error);
                }

                errorsToBeSummed.add(errors);
            }
            neuralLayer.setErrors(sumErrors(errorsToBeSummed));

            prevLayer = neuralLayer;
        }
    }

    private List<Double> sumErrors(List<List<Double>> list) {
        int length = list.get(0).size();
        List<Double> res = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            double sum = 0;
            for (int j = 0; j < list.size(); j++) {
                sum += list.get(j).get(i);
            }
            res.add(sum);
        }
        return res;
    }

    private void validateInput() {
        if (layerSizes.size() < 2) {
            String message = "2 or more layers are needed, give the constructor more inputs";
            throw new RuntimeException(message);
        }

        for (int i : layerSizes) {
            if (i < 1) {
                String message = "The inputs can't be less than 1";
                throw new RuntimeException(message);
            }
        }
    }

    private int indexOfMax(List<Double> list) {
        double max = 0.0;
        int index = 0;

        for(int i = 0; i < list.size(); i++){
            double item = list.get(i);
            if(item > max){
                max = item;
                index = i;
            }
        }

        return index;
    }

    private List<Double> softmax(List<Double> vector){
        double vectorSum = vector.stream().mapToDouble(x -> x).sum();
        List<Double> res = new ArrayList<>();
        for(double d : vector){
            double softmaxValue = d / vectorSum;
            res.add(softmaxValue);
        }
        return res;
    }
}
