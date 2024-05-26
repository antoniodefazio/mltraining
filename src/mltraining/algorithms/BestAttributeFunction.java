package mltraining.algorithms;

import java.util.List;
import java.util.Map;

@FunctionalInterface
public interface BestAttributeFunction {

	String chooseBestAttribute(List<Map<String, String>> data, List<String> attributes, String targetAttribute);

}
