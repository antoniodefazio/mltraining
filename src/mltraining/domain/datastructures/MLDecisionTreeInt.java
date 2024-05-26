package mltraining.domain.datastructures;

import java.util.List;
import java.util.Map;

public interface MLDecisionTreeInt {

	String classify(Map<String, String> instance);

	void train(List<Map<String, String>> data, List<String> attributes, String targetAttribute);
}
