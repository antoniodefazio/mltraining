package mltraining.inductive;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltraining.algorithms.BestAttributeFunction;
import mltraining.algorithms.ID3;
import mltraining.algorithms.PartitionerImpl;
import mltraining.domain.TrainingData;
import mltraining.domain.TrainingDataConcept;
import mltraining.domain.datastructures.MLDecisionTreeImpl;
import mltraining.domain.datastructures.Outlook;
import mltraining.utility.MLPrinter;

public class LearningProblem {

	public static void main(String[] args) throws FileNotFoundException, IOException {
		final Boolean[][] boolTrainingData = { { false, false, false, false, false },
				{ false, false, false, true, true }, { false, false, true, false, false },
				{ false, false, true, true, false }, { false, true, false, false, false },
				{ false, true, false, true, true }, { false, true, true, false, false },
				{ false, true, true, true, false }, { true, false, false, false, false } };
		final TrainingDataConcept examples = new TrainingDataConcept();

		for (final Boolean[] row : boolTrainingData) {
			final Map<String, Boolean> tuple = new HashMap<>();
			int boolIndex = 0;
			for (final Boolean val : row) {
				tuple.put("x" + (boolIndex + 1), val);

				boolIndex++;
			}
			examples.addTraining(tuple);
		}

		// Print the truth table
		// MLPrinter.printTrainingData(examples, "x5");
		final TrainingData<Integer> integerTrainingData = new TrainingData<>();
		final Integer[] source = new Integer[] { 34, 56, 78 };

		final Map<String, Integer> tuple = new HashMap<>();
		int tupleIdex = 0;
		for (final Integer val : source) {
			tuple.put("x" + (tupleIdex + 1), val);

			tupleIdex++;
		}
		integerTrainingData.addTraining(tuple);

		// MLPrinter.printTrainingData(integerTrainingData, "x5");
		final List<Map<String, String>> data = new ArrayList<>();
		// Classify a new instance
		final Map<String, String> instance = Map.of(Outlook.class.getSimpleName(), Outlook.Rain.name(), "Temperature",
				"Hot", "Humidity", "Normal", "Windy", "Weak"); // leaf 1
		final BestAttributeFunction bestAttributeFunction =
				/*
				 * (dataList, attributes, targetAttribute) -> { if
				 * (attributes.contains(OUTLOOK)) { return OUTLOOK; } if
				 * (attributes.contains(WINDY)) { return WINDY; } return attributes.get(0); };
				 */
				new ID3(PartitionerImpl.singletonReference);
		final MLDecisionTreeImpl tree = new MLDecisionTreeImpl(bestAttributeFunction,
				PartitionerImpl.singletonReference);
		// Attributes to consider
		final List<String> xAttributes = List.of(Outlook.class.getSimpleName(), "Temperature", "Humidity", "Windy");

		// Target attribute
		final String targetAttribute = "PlayTennis";

		data.add(Map.of(Outlook.class.getSimpleName(), Outlook.Sunny.name(), "Temperature", "Hot", "Humidity", "High",
				"Windy", "Weak", "PlayTennis", "No")); // leaf 0
		final TrainingData<String> tennisTrainingData = new TrainingData<>();
		tennisTrainingData.setTrainings(data);
		// Create and train the decision tree
		MLPrinter.printTrainingData(tennisTrainingData, targetAttribute);
		tree.train(data, xAttributes, targetAttribute);

		System.out.println("******** ");
		// Print the decision tree
		System.out.println(tree);
		String result = tree.classify(instance);
		System.out.println("Classification: of instance " + instance + " with result " + result);
		final Map<String, String> newData = Map.of(Outlook.class.getSimpleName(), Outlook.Overcast.name(),
				"Temperature", "Hot", "Humidity", "High", "Windy", "Weak", "PlayTennis", "Yes");
		System.out.println("Insert new data " + newData);
		data.add(newData); // leaf 2
		MLPrinter.printTrainingData(tennisTrainingData, targetAttribute);
		tree.train(data, xAttributes, targetAttribute);
		// Print the decision tree
		System.out.println(tree);
		result = tree.classify(instance);
		System.out.println("Classification: of instance " + instance + " with result " + result);
		data.add(Map.of(Outlook.class.getSimpleName(), Outlook.Sunny.name(), "Temperature", "Hot", "Humidity", "Normal",
				"Windy", "Strong", "PlayTennis", "Yes")); // leaf 1
		tree.train(data, xAttributes, targetAttribute);
		// Print the decision tree
		System.out.println(tree);
		result = tree.classify(instance);
		System.out.println("Classification: of instance " + instance + " with result " + result);
		data.add(Map.of(Outlook.class.getSimpleName(), Outlook.Sunny.name(), "Temperature", "Hot", "Humidity", "Normal",
				"Windy", "Weak", "PlayTennis", "Yes")); // leaf 1
		MLPrinter.printTrainingData(tennisTrainingData, targetAttribute);
		tree.train(data, xAttributes, targetAttribute);
		// Print the decision tree
		System.out.println(tree);
		result = tree.classify(instance);
		System.out.println("Classification: of instance " + instance + " with result " + result);
		data.add(Map.of(Outlook.class.getSimpleName(), Outlook.Rain.name(), "Temperature", "Hot", "Humidity", "Normal",
				"Windy", "Strong", "PlayTennis", "No")); // leaf 3
		MLPrinter.printTrainingData(tennisTrainingData, targetAttribute);
		tree.train(data, xAttributes, targetAttribute);
		// Print the decision tree
		System.out.println(tree);
		result = tree.classify(instance);
		System.out.println("Classification: of instance " + instance + " with result " + result);
		data.add(Map.of(Outlook.class.getSimpleName(), Outlook.Rain.name(), "Temperature", "Hot", "Humidity", "Normal",
				"Windy", "Weak", "PlayTennis", "Yes")); // leaf 4
		System.out.println(tree);// withour train
		tree.train(data, xAttributes, targetAttribute);
		System.out.println(tree);// after train
		result = tree.classify(instance);
		/*
		 * ^^^^^^^^^^^^^^Bank Loan^^^^^^^^^
		 */

		final List<List<String>> records = new ArrayList<>();
		int countRow = 0;
		final int MAX_ROW = 400;
		final BufferedReader br = new BufferedReader(new FileReader("loan_data.csv"));
		String line;
		while ((line = br.readLine()) != null) {

			final String[] values = line.split(",");
			// System.out.println("values" + values);
			if (countRow > 0) {
				records.add(Arrays.asList(values));
			}
			if (countRow > MAX_ROW) {
				break;
			}
			countRow++;
		}
		br.close();
		final String[] allLoanAttr = new String[] { "credit.policy", "purpose", "int.rate", "installment",
				"log.annual.inc", "dti", "fico", "days.with.cr.line", "revol.bal", "revol.util", "inq.last.6mths",
				"delinq.2yrs", "pub.rec", "not.fully.paid" };
		final String loanTargetAttribute = "not.fully.paid";
		final String[] loanAttributesXArr = Arrays.copyOfRange(allLoanAttr, 0, allLoanAttr.length - 1);
		final List<String> loanAttributesX = Arrays.asList(loanAttributesXArr);
		final List<Map<String, String>> loanData = new ArrayList<>();
		System.out.println("records" + records);
		System.out.println("loanAttributesX" + loanAttributesX);
		for (final List<String> loanTuple : records) {
			final Map<String, String> loanTupleMap = new HashMap<>();
			for (int i = 0; i < allLoanAttr.length; i++) {
				loanTupleMap.put(allLoanAttr[i], valueInRange(allLoanAttr[i], loanTuple.get(i)));
			}
			loanData.add(loanTupleMap);
		}
		final TrainingData<String> loanTrainingData = new TrainingData<>();
		loanTrainingData.setTrainings(loanData);
		MLPrinter.printTrainingData(loanTrainingData, loanTargetAttribute);
		tree.train(loanData, loanAttributesX, loanTargetAttribute);
		System.out.println(tree);// after train
		result = tree.classify(loanData.get(10));
		System.out.println("Classification: of instance " + loanData.get(10) + " with result " + result);

	}

	private static final String valueInRange(String attribute, String value) {
		Double doubleVal = null;
		try {
			doubleVal = Double.parseDouble(value);
		} catch (final NumberFormatException nfe) {
			// not a double
		}
		if ("int.rate".equals(attribute)) {
			doubleVal *= 100;
			if (doubleVal < 5) {
				return "Optimum";
			}
			if (doubleVal < 8) {
				return "Good";
			}
			if (doubleVal < 11) {
				return "Normal";
			}
			if (doubleVal < 14) {
				return "High";
			}
			return "SoHigh";
		}
		if ("installment".equals(attribute)) {
			if (doubleVal < 200) {
				return "Under200";
			}
			if (doubleVal < 400) {
				return "B200A400";
			}
			if (doubleVal < 600) {
				return "B400A600";
			}
			if (doubleVal < 800) {
				return "B600A800";
			}
			return "O800";
		}
		if ("log.annual.inc".equals(attribute)) {
			if (doubleVal < 5) {
				return "LogOptimum";
			}
			if (doubleVal < 8) {
				return "LogGood";
			}
			if (doubleVal < 11) {
				return "LogNormal";
			}
			if (doubleVal < 14) {
				return "LogHigh";
			}
			return "LogSoHigh";
		}
		if ("dti".equals(attribute)) {
			if (doubleVal < 5) {
				return "DtiOptimum";
			}
			if (doubleVal < 8) {
				return "DtiGood";
			}
			if (doubleVal < 11) {
				return "DtiNormal";
			}
			if (doubleVal < 14) {
				return "DtiHigh";
			}
			return "SoHigh";
		}
		if ("fico".equals(attribute)) {
			if (doubleVal < 200) {
				return "FicoUnder200";
			}
			if (doubleVal < 400) {
				return "FicoB200A400";
			}
			if (doubleVal < 600) {
				return "FicoB400A600";
			}
			if (doubleVal < 800) {
				return "FicoB600A800";
			}
			return "FicoO800";
		}
		if ("days.with.cr.line".equals(attribute)) {
			if (doubleVal < 1500) {
				return "DaysUnder5Y";
			}
			if (doubleVal < 3000) {
				return "DaysUnder10Y";
			}
			if (doubleVal < 4500) {
				return "DaysUnder15Y";
			}
			if (doubleVal < 6000) {
				return "DaysUnder20Y";
			}
			return "DaysOver20Y";
		}
		if ("revol.bal".equals(attribute)) {
			if (doubleVal < 20000) {
				return "RBUnder20000";
			}
			if (doubleVal < 40000) {
				return "RBUnder200A40000";
			}
			if (doubleVal < 60000) {
				return "RBUnder400A60000";
			}
			if (doubleVal < 80000) {
				return "RBUnder600A80000";
			}
			return "RBOver80000";
		}
		if ("revol.util".equals(attribute)) {
			if (doubleVal < 20) {
				return "RVUnder200";
			}
			if (doubleVal < 40) {
				return "RVB200A400";
			}
			if (doubleVal < 60) {
				return "RVB400A600";
			}
			if (doubleVal < 80) {
				return "RVB600A800";
			}
			return "RVO800";
		}
		if ("not.fully.paid".equals(attribute)) {
			return doubleVal == 0 ? "PAID" : "NOTPAID";
		}
		return value;
	}
}
