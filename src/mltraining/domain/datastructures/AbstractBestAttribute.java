package mltraining.domain.datastructures;

import mltraining.algorithms.BestAttributeFunction;
import mltraining.algorithms.PartitionerInt;

public abstract class AbstractBestAttribute implements BestAttributeFunction {

	private final PartitionerInt partitioner;

	protected AbstractBestAttribute(PartitionerInt partitioner) {
		this.partitioner = partitioner;
	}

	public PartitionerInt getPartitioner() {
		return partitioner;
	}

}
