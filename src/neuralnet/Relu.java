package neuralnet;

public class Relu implements ActivationFunction {
    @Override
    public double calc(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivative(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    }

}
