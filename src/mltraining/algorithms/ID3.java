package mltraining.algorithms;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import mltraining.domain.datastructures.AbstractBestAttribute;

public class ID3 extends AbstractBestAttribute {

	private static final Logger LOGGER = Logger.getLogger(ID3.class.getName());

	public ID3(PartitionerInt partitioner) {
		super(partitioner);
	}

	private double calculateShannonEntropy(List<Map<String, String>> data, String targetAttribute) {
		// group all data by class
		final Map<String, Integer> classCounts = new HashMap<>();
		for (final Map<String, String> instance : data) {
			final String classs = instance.get(targetAttribute);
			classCounts.put(classs, classCounts.getOrDefault(classs, 0) + 1);
		}
		LOGGER.info(String.format(" Counter of all data grouped by targetAttribute's value %s=%s ", targetAttribute,
				classCounts));

		double entropy = 0;
		for (final Map.Entry<String, Integer> entry : classCounts.entrySet()) {
			final double classValueProbability = (double) entry.getValue() / data.size();
			LOGGER.info(String.format(" Probability of %s=%s ", entry.getKey(), classValueProbability));
			final double log2ClassValueProbability = Math.log(classValueProbability) / Math.log(2);
			entropy -= classValueProbability * log2ClassValueProbability;
		}
		LOGGER.info(String.format(" Entropy of attribute  %s=%s ", targetAttribute, entropy));
		return entropy;
	}

	@Override
	public String chooseBestAttribute(List<Map<String, String>> data, List<String> attributes, String targetAttribute) {
		final double baseEntropy = calculateShannonEntropy(data, targetAttribute);
		LOGGER.info(String.format(" Base Entropy H(S) of attribute  %s=%s ", targetAttribute, baseEntropy));
		double bestGain = Double.MIN_VALUE;
		String bestAttribute = null;

		for (final String attribute : attributes) {
			final double attributeAverageInformation = gainAfterSplitting(data, targetAttribute, attribute);
			LOGGER.info(String.format(" Subset Entropy H(S|%s)=%s ", attribute, attributeAverageInformation));
			final double infoGain = baseEntropy - attributeAverageInformation;
			LOGGER.info(String.format(" Info Gain IG(S|%s)=%s ", attribute, infoGain));
			if (infoGain > bestGain) {
				bestGain = infoGain;
				bestAttribute = attribute;
			}
		}
		return bestAttribute;
	}

	private double gainAfterSplitting(List<Map<String, String>> data, String targetAttribute, final String attribute) {
		double afterSplitGain = 0;
		final Map<String, List<Map<String, String>>> partitions = getPartitioner().partitionData(data, attribute);
		for (final Map.Entry<String, List<Map<String, String>>> subset : partitions.entrySet()) {
			final List<Map<String, String>> splittedSet = subset.getValue();
			final double attributeValueProbability = (double) splittedSet.size() / data.size();
			LOGGER.info(String.format(" After split Probability of %s=%s ", attribute, attributeValueProbability));

			afterSplitGain += attributeValueProbability * calculateShannonEntropy(splittedSet, targetAttribute);
		}
		LOGGER.info(String.format(" afterSplitGain of %s=%s ", targetAttribute, afterSplitGain));
		return afterSplitGain;
	}

}
