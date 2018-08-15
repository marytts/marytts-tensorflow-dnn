package marytts.dnn;

import marytts.data.SupportedSequenceType;

/**
 *
 *
 * @author <a href="mailto:slemaguer@coli.uni-saarland.de">Sébastien Le Maguer</a>
 */
public class TensorflowSupportedSequenceType
{

    public static final String NORMALISED_FEATURES = "NORMALISED_FEATURES";

    static {
        SupportedSequenceType.addSupportedType(NORMALISED_FEATURES);
    }
}
