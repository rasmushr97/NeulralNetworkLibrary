package neuralnet;

import java.util.ArrayList;
import java.util.List;

public class HiddenLayer {
    private List<Neuron> neurons;
    private List<Double> output;
    private int prevLayerSize;

    HiddenLayer(ActivationFunction activationFunction, int prevLayerSize, int layerSize) {
        this.prevLayerSize = prevLayerSize;

        neurons = new ArrayList<>();
        for (int i = 0; i < layerSize; i++) {
            Neuron neuron = new Neuron(prevLayerSize, activationFunction);
            neurons.add(neuron);
        }
    }

    void setErrors(List<Double> errors) {
        if(errors.size() != neurons.size()){
            throw new RuntimeException("Not the same amount of errors as neurons");
        }

        for(int i = 0; i < errors.size(); i++){
            neurons.get(i).setError(errors.get(i));
        }
    }

    List<Neuron> getNeurons() {
        return neurons;
    }

    void setInput(List<Double> input) {
        if (input.size() != prevLayerSize) {
            throw new RuntimeException("Input does not match the size of the previous layer");
        }

        for(Neuron neuron : neurons){
            neuron.setInput(input);
        }
    }

    void calc() {
        output = new ArrayList<>();
        for (Neuron neuron : neurons) {
            neuron.calc();
            double neuronOutput = neuron.getOutput();
            output.add(neuronOutput);
        }
    }

    List<Double> getOutput() {
        return output;
    }

    void performBackprop() {

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            neuron.performBackprop();
        }
    }
}
