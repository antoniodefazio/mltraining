package mltraining.utility;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import mltraining.domain.TrainingData;

public interface MLPrinter {

	static <T> void printTrainingData(TrainingData<T> trainings, String classlabel) {
		final StringBuilder domain = new StringBuilder();

		final StringBuilder header = new StringBuilder();
		for (final String attribute : trainings.allAttributesInFirstRow()) {
			if (!attribute.equals(classlabel)) {
				domain.append(attribute + " ");

				header.append("-----");
			}
		}
		domain.append(" |  " + classlabel + " ");
		System.out.println(domain);
		System.out.println(header);
		final StringBuilder examples = new StringBuilder();

		for (int i = 0; i < trainings.allAttributesInFirstRow().size() - 1; i++) {
			examples.append("%s  ");
		}
		examples.append(" |  %s%n");
		for (final Map<String, T> tuple : trainings) {

			System.out.printf(examples.toString(), toObjectArray(tuple));
		}
	}

	static <T> Object[] toObjectArray(Map<String, T> row) {

		if (row.entrySet().iterator().next().getValue() instanceof Boolean) {
			final Integer[] array = new Integer[row.size()];
			int i = 0;
			for (final Map.Entry<String, T> entry : row.entrySet()) {
				if (Boolean.TRUE.equals(entry.getValue())) {
					array[i] = 1;
				} else {
					array[i] = 0;
				}
				i++;
			}
			return array;
		}
		final List<T> list = new ArrayList<>();
		for (final String key : row.keySet()) {
			list.add(row.get(key));
		}

		return list.toArray();
	}

}
