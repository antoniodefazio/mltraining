package mltraining.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PartitionerImpl implements PartitionerInt {

	public static PartitionerInt singletonReference = new PartitionerImpl();

	private PartitionerImpl() {

	}

	@Override
	public Map<String, List<Map<String, String>>> partitionData(List<Map<String, String>> data, String attribute) {
		final Map<String, List<Map<String, String>>> partitions = new HashMap<>();
		for (final Map<String, String> instance : data) {
			final String valueOfAttribute = instance.get(attribute);
			partitions.computeIfAbsent(valueOfAttribute, k -> new ArrayList<>()).add(instance);
		}
		return partitions;
	}

}
