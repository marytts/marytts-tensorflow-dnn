package marytts.dnn;


import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.tensorflow.Tensor;
import org.testng.annotations.Test;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import marytts.dnn.features.FeatureChunk;

/**
 *
 *
 * @author <a href="mailto:slemaguer@coli.uni-saarland.de">Sébastien Le Maguer</a>
 */
public class TestTensor
{


    @Test
    public void testTensorDoubleConversion() {
        double[][] ref = {{0.29219939, 0.50895349},
                          {0.65312174, 0.75139362},
                          {0.82279594, 0.87822527}};
        DenseDoubleMatrix2D data = new DenseDoubleMatrix2D(ref);
        FeatureChunk fc = new FeatureChunk(data);
        Tensor<Float> ten = fc.getTensorData();
        FeatureChunk fc2 = new FeatureChunk(ten, null);
        double[][] test = fc2.getArrayValues();
        for (int t=0; t<ref.length; t++)
            assertThat(test[t]).containsExactly(ref[t], within(1e-6));
    }
}
