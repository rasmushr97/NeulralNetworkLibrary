package neuralnet;

public class Sigmoid implements ActivationFunction {
    @Override
    public double calc(double x) {
        double Ex = Math.pow(Math.E, x);
        return Ex/(Ex + 1);
    }

    @Override
    public double derivative(double x) {
        return x * (1 - x);
    }
}