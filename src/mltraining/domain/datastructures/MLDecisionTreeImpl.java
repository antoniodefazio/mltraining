package mltraining.domain.datastructures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import mltraining.algorithms.BestAttributeFunction;
import mltraining.algorithms.PartitionerInt;

public class MLDecisionTreeImpl implements MLDecisionTreeInt {

	private static final Logger LOGGER = Logger.getLogger(MLDecisionTreeImpl.class.getName());

	private MLDecisionNode root;
	// The lambda function is used to change the algorithm
	private final BestAttributeFunction bestAttributeFunction;

	private final PartitionerInt partitionerInt;

	private final String INDENTATOR = "|__ ";

	public MLDecisionTreeImpl(BestAttributeFunction bestAttributeFunction, PartitionerInt partitionerInt) {
		this.bestAttributeFunction = bestAttributeFunction;
		this.partitionerInt = partitionerInt;
	}

	private MLDecisionNode buildTree(List<Map<String, String>> data, List<String> attributes, String targetAttribute) {

		final String allSameClassLabel = checkAndGetEveryItemSameClass(data, targetAttribute);
		LOGGER.info(String.format(" Considered attributes %s . allSameClassLabel of targetAttribute  %s=%s ",
				attributes, targetAttribute, allSameClassLabel));
		// If all the data has the same target attribute value, return a leaf node with
		// that value
		if (allSameClassLabel != null) {
			// base case of the recursive function
			return MLDecisionNode.builder().leaf(allSameClassLabel).build(); // so return the class label
		}

		// If no more attributes to split on, return a leaf node with the majority label
		if (attributes.isEmpty()) {
			final String majorityLabel = getMajorityLabel(data, targetAttribute);
			LOGGER.info(
					String.format(" attributes.isEmpty() of getMajorityLabel  %s=%s ", targetAttribute, majorityLabel));
			// base case of the recursive function
			return MLDecisionNode.builder().leaf(majorityLabel).build();
		}

		// Select the best attribute to split on
		final String bestAttribute = bestAttributeFunction.chooseBestAttribute(data, attributes, targetAttribute);
		LOGGER.info(String.format(" bestAttribute of set %s with targetAttribute  %s=%s ", attributes, targetAttribute,
				bestAttribute));
		final MLDecisionNode innerRoot = MLDecisionNode.builder().decisionNodewithAttribute(bestAttribute).build();

		final Map<String, List<Map<String, String>>> partitionsFromBestAttribute = partitionerInt.partitionData(data,
				bestAttribute);

		LOGGER.info(String.format(" partitions based on targetAttribute  %s=%s ", bestAttribute,
				partitionsFromBestAttribute));

		// For each value of the best attribute, create a subtree recursively with
		// remainingAttributes
		for (final Map.Entry<String, List<Map<String, String>>> partitionEntry : partitionsFromBestAttribute
				.entrySet()) {
			final String valueOfBestAttribute = partitionEntry.getKey();
			final List<Map<String, String>> dataWithSameValueOfBestAttribute = partitionEntry.getValue();
			if (dataWithSameValueOfBestAttribute.isEmpty()) {
				// base case of the recursive function
				LOGGER.info(String.format(" Creating a leaf for %s with targetAttribute %s ", valueOfBestAttribute,
						targetAttribute));
				innerRoot.getChildren().put(valueOfBestAttribute,
						MLDecisionNode.builder().leaf(getMajorityLabel(data, targetAttribute)).build());
			} else

			{
				final List<String> remainingAttributes = new ArrayList<>(attributes);
				remainingAttributes.remove(bestAttribute);
				innerRoot.getChildren().put(valueOfBestAttribute,
						buildTree(dataWithSameValueOfBestAttribute, remainingAttributes, targetAttribute));
			}
		}

		return innerRoot;
	}

	private String checkAndGetEveryItemSameClass(List<Map<String, String>> data, String targetAttribute) {
		final String firstLabel = data.get(0).get(targetAttribute);
		for (final Map<String, String> instance : data) {
			if (!instance.get(targetAttribute).equals(firstLabel)) {
				return null;
			}
		}
		return firstLabel;
	}

	// Recursive method to classify a new instance using the trained decision tree
	@Override
	public String classify(Map<String, String> instance) {
		return classifyRecursive(instance, root);
	}

	private String classifyRecursive(Map<String, String> instance, MLDecisionNode node) {
		LOGGER.info(String.format(" Classifying instance %s on Node %s root %s", instance, node, node == root));

		if (node.isLeaf()) {
			final String classs = node.getLabel();
			LOGGER.info(String.format(" Classifying Node %s is leaf so instance %s is %s", node, instance, classs));
			// Leaf node
			// base case of the recursive function
			return classs;
		}

		final String attributeValue = instance.get(node.getAttribute());
		final MLDecisionNode childNode = node.getChildren().get(attributeValue);
		if (childNode == null) {
			LOGGER.info(String.format(
					" childNode == null so instance %s is Unknown instance, value not present in training data ",
					instance));
			// Unknown instance, value not present in training data
			// base case of the recursive function
			return null;
		}
		// Recursively classify
		return classifyRecursive(instance, childNode);
	}

	private String getMajorityLabel(List<Map<String, String>> data, String targetAttribute) {
		final Map<String, Integer> countTargetAttributeValues = new HashMap<>();
		for (final Map<String, String> instance : data) {
			final String label = instance.get(targetAttribute);
			countTargetAttributeValues.put(label, countTargetAttributeValues.getOrDefault(label, 0) + 1);
		}

		String majorityLabel = null;
		int maxCount = 0;
		for (final Map.Entry<String, Integer> entry : countTargetAttributeValues.entrySet()) {
			if (entry.getValue() > maxCount) {
				maxCount = entry.getValue();
				majorityLabel = entry.getKey();
			}
		}
		return majorityLabel;
	}

	// Method to return the tree as a string
	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		toString(sb, root, "");
		return sb.toString();
	}

	// Recursive helper method to build the string representation of the tree
	private void toString(StringBuilder sb, MLDecisionNode node, String prefix) {
		if (node.isLeaf()) {
			// base case of the recursive function
			sb.append(prefix).append(INDENTATOR).append("Label: ").append(node.getLabel()).append("\n");
		} else {

			sb.append(prefix).append(INDENTATOR).append("Attribute: ").append(node.getAttribute()).append("\n");

			for (final Map.Entry<String, MLDecisionNode> entry : node.getChildren().entrySet()) {
				sb.append(prefix).append("    ").append(INDENTATOR).append("Value: ").append(entry.getKey())
						.append("\n");

				toString(sb, entry.getValue(), prefix + "    ");
			}
		}
	}

	// Method to train the decision tree with the given data
	@Override
	public void train(List<Map<String, String>> data, List<String> attributes, String targetAttribute) {
		root = buildTree(data, attributes, targetAttribute);
	}

}
