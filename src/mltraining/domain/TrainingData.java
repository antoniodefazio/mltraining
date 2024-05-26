package mltraining.domain;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TrainingData<T> implements Iterable<Map<String, T>> {

	private List<Map<String, T>> trainings = new ArrayList<>();

	public void addTraining(Map<String, T> example) {
		this.trainings.add(example);
	}

	public Set<String> allAttributesInFirstRow() {
		return trainings.isEmpty() ? Set.of() : trainings.get(0).keySet();
	}

	public List<Map<String, T>> getTrainings() {
		return trainings;
	}

	@Override
	public Iterator<Map<String, T>> iterator() {

		return trainings.iterator();
	}

	public void setTrainings(List<Map<String, T>> examples) {
		this.trainings = examples;
	}
}
