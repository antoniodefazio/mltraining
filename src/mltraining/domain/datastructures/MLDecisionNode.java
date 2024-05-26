package mltraining.domain.datastructures;

import java.util.HashMap;
import java.util.Map;

public class MLDecisionNode {
	public static final class Builder {
		private String attribute;
		private String label;
		private Map<String, MLDecisionNode> children = new HashMap<>();

		private Builder() {
		}

		public MLDecisionNode build() {
			return new MLDecisionNode(this);
		}

		public Builder decisionNodewithAttribute(String attribute) {
			this.attribute = attribute;
			return this;
		}

		public Builder leaf(String label) {
			this.label = label;
			return this;
		}

		public Builder withChildren(Map<String, MLDecisionNode> children) {
			this.children = children;
			return this;
		}
	}

	public static Builder builder() {
		return new Builder();
	}

	private final String attribute;

	private final String label;

	private Map<String, MLDecisionNode> children = new HashMap<>();

	private MLDecisionNode(Builder builder) {
		this.attribute = builder.attribute;
		this.label = builder.label;
		this.children = builder.children;
	}

	public String getAttribute() {
		return attribute;
	}

	public Map<String, MLDecisionNode> getChildren() {
		return children;
	}

	public String getLabel() {
		return label;
	}

	public boolean isLeaf() {
		return label != null;
	}

	public void setChildren(Map<String, MLDecisionNode> children) {
		this.children = children;
	}

	@Override
	public String toString() {
		return "MLTreeNode [attribute=" + attribute + ", label=" + label + "]";
	}

}
