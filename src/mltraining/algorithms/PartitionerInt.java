package mltraining.algorithms;

import java.util.List;
import java.util.Map;

@FunctionalInterface
public interface PartitionerInt {

	Map<String, List<Map<String, String>>> partitionData(List<Map<String, String>> data, String attribute);

}
