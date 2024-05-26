package mltraining.domain;

import java.util.HashMap;
import java.util.Map;

public class TrainingDataConcept extends TrainingData<Boolean> {

	public void addTriningFromInt(int[] example) {
		final Boolean[] exampleBool = new Boolean[example.length];
		for (int i = 0; i < example.length; i++) {
			if (example[i] != 0) {
				exampleBool[i] = true;
			}
		}
		final Map<String, Boolean> tuple = new HashMap<>();
		for (int i = 0; i < example.length; i++) {
			tuple.put("x" + (i + 1), exampleBool[i]);
		}
		addTraining(tuple);
	}

}
