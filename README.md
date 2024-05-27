# How to create from scratch a Machine Learning Decisional Tree in Java from training data. Concrete example of bank loan payment forecast

### Decision Tree Representation

The decision tree can be considered not only an alternative and clearer view of the data, but also a useful classification tool and possibly prediction of the class of a data item.

The project I propose in the article shows the algorithm for generating the decision tree based on the training data.
The program aims to generate the classifier for any set of attributes and data, therefore it generates an n-ary decision tree for a target attribute that can have n labels.

The example considered in the main method( class https://github.com/antoniodefazio/mltraining/blob/master/src/mltraining/inductive/LearningProblem.java) is the classic one of the tennis match which is played based on some weather conditions, which decision tree is: 

![tree](https://github.com/antoniodefazio/mltraining/assets/61966052/947cfefa-04eb-46d6-b4b4-c5f743c6c98b)

The tree is therefore based on existing data, so built by training from the data, and can also predict, and classify a tuple of data such as the following:

_(Outlook = Sunny, Temperature = Hot, Humidity = High, Wind = Strong)_
 would be classified as a negative instance (i.e., the tree predicts that PlayTennis = no).
This decision tree, since the range of the target attribute is binary (Yes or No), it can alternatively be represented by the following boolean expression

_Outlook = Sunny AND Humidity = Normal OR Outlook = Overcast OR Outlook = Rain  AND Wind = Weak_


### In general, how to build the tree starting from the training data?

We will use the ID3 algorithm first by making a discursive description of it, then pseudocode and finally in Java. This is the wiki link https://en.wikipedia.org/wiki/ID3_algorithm. 
 
The most important part of the algorithm is to establish on a statistical basis which is the __best attribute__ to use as the root of the tree, for obvious reasons, later a descendant of the root node is then created for each possible value of this attribute, and so on recursively, so the entire process is then repeated using the training examples associated with each descendant node to select the best attribute to test at that point in the tree.

Here is my starting point, the pseudcode function, with 3 parameters, to generate the tree(pay attention to indentation):

    ID3(examples, targetattribute, attributes)
    
    Parameter 1: __examples__, the training examples.
    
    Parameter 2: __targetattribute__, the attribute whose value is to be predicted by the tree.
    
    Parameter 3: __attributes__, a list of other attributes that may be tested by the learned decision tree. 

     If all __examples__ are of the same class, return a single-node tree Root, with label = class

     If __attributes__ is empty, Return a single-node tree Root, with label = most common value of
     __targetattribute__ in __examples__


    Let B the attribute from __attributes__ that best classifies __examples__

    Create a Root decision node which attribute is B

    For each possible value vi of B

        Add to this node a new tree branch corresponding to the test B = vi. So let __examplesBest__ be the subset of __examples__ that have value vi for B(partition)

        If __examplesBest__ is empty

            then below this node add a new branch with leaf node with label = most common value of the targetattribute in __examples__

            else below this node add new branch with subtree ID3(__examplesBest__, targetattribute, attributes – (B)))
            
     Return the Root decision node

Full Java code of ID3 at https://github.com/antoniodefazio/mltraining/blob/master/src/mltraining/domain/datastructures/MLDecisionTreeImpl.java method _buildTree_

ID3's philosophy is that shorter trees are preferred on longer trees, and which places relevant information close to the root are preferred over, therefore the algorithm always tries to put the attributes that have the greatest correlation with the target attribute at the top.

### Which Attribute Is the Best Classifier?

The best attribute as root is obviously the one that has the greatest impact, therefore high correlation, with the target attribute, so it is the best __"separator"__(during split) of training data. What is a good quantitative measure of the worth of an attribute? A statistical property, called __information gain__, that measures how well a given attribute separates the training examples according to their target classification. ID3 uses this information gain measure to select among the candidate attributes at each step while growing the tree.
So in order to better understand how to establish the best attribute during split we have to understand the concepts of information gain and __entropy__ precisely. We in fact begin by defining a measure commonly used in information theory, called entropy, its name comes from one of the fathers of information theory, Claude Shannon. I had studied this formula at University for the "Theory of Waves" exam. 

_Legend has it that the terms information gain and entropy are purposely designed to cause confusion! In fact it was John von Neumann suggested that Shannon use the term entropy because people wouldn't know what it meant. Claude Shannon is considered one of the most intelligent people of the twentieth century.
It was said of Claude Shannon:“There were many at Bell Labs and MIT who compared Shannon's intuition to that of Einstein. Others found that comparison unfair, unfair to Shannon”..._

Entropy is defined as the __expected value of information__.



In ML context Entropy is the expected value of the information __relating to the classes__. 

For n classes of data 

n-class Entropy -> _E(S) = ∑ -(pᵢ*log₂pᵢ)_ 

So given a collection S, containing positive and negative examples(2 classes) of some target concept, the entropy of S relative to this boolean classification is

2-class Entropy -> _E(S) =-(p₁ * log₂p₁ + p₂ * log₂p₂)_ 

where  p₁ is the probability of  positive examples while p₂ negative ones.

ID3 leverages the information gain IG measure of to select the best attribute at each step of tree construction, this information is closely related to the entropy mentioned above. IG represents the expected reduction in entropy caused by partitioning (after dividing) data based on
this attribute. More precisely, the information gain, Gain(S, A) of an attribute A is:


_Gain(S,A) = E(S) – E(S | A)_

where E(S) is the above entropy before split, while E(S|A) is the entropy given A, means the entropy after splitting based on A.

Fill Java code for Entropy and Infomration Gain at https://github.com/antoniodefazio/mltraining/blob/master/src/mltraining/algorithms/ID3.java

The part relating to choosing the best attribute during the split phase can be changed as I use a Java Lambda Function.

In the main method( class https://github.com/antoniodefazio/mltraining/blob/master/src/mltraining/inductive/LearningProblem.java ) the tree for the classic example of the tennis match is generated.

# BANK LOAN

The final part of the main method( class https://github.com/antoniodefazio/mltraining/blob/master/src/mltraining/inductive/LearningProblem.java)  is very interesting because the decision tree for a bank loan is generated on the basis of the csv indicating the data relating to whether the bank loan has been paid in full or not. In fact, the data contains the target attribute __“not.fully.paid”__ which classifies all the examples of the csv. The other variables meaning is:

_credit.policy_: 1 if the customer meets the credit underwriting criteria of LendingClub.com , and 0 otherwise.

_purpose_: The purpose of the loan (takes values "credit_card", "debt_consolidation", " educational", "major_purchase", "small_business", and "all_other").

_int.rate_: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0. 11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.

_installment_: The monthly installments owed by the borrower if the loan is funded.

_log.annual.inc_: The natural log of the self-reported annual income of the borrower.

_dti_: The debt-to-income ratio of the borrower (amount of debt divided by annual income).

_fico_: The FICO credit score of the borrower.

_days.with.cr.line_: The number of days the borrower has had a credit line.

_revol.bal_: The borrower’s revolvin balance (amount unpaid at the end of the credit card billing cycle).

_revol.util_: The borrower’s revolving line utilization rate (the amount of the credit line used relative to total credit available).

_inq.last.6mths_: The borrower’s number of inquiries by creditors in the last 6 months.

_delinq.2yrs_: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.

_pub.rec_: The borrower’s number of derogatory public records (bankruptcy filings, tax liens, or judgments).

Obviously I discretized the "continuous" variables in my own way, creating arbitrary ranges.

I really like to represent the tree with "my" ToString, test it locally!!

____Attribute: int.rate
___|____Value: High
___|____Attribute: purpose
___|___|____Value: debt_consolidation
___|___|____Attribute: revol.bal
___|___|___|____Value: RBUnder400A60000
___|___|___|____Label: PAID
___|___|___|____Value: RBUnder200A40000
___|___|___|____Attribute: installment
___|___|___|___|____Value: B600A800
___|___|___|___|____Label: PAID
___|___|___|___|____Value: Under200
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: B200A400
___|___|___|___|____Attribute: dti
___|___|___|___|___|____Value: DtiGood
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: SoHigh
___|___|___|___|___|____Label: PAID
___|___|___|___|____Value: B400A600
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: O800
___|___|___|___|____Label: PAID
___|___|___|____Value: RBOver80000
___|___|___|____Label: PAID
___|___|___|____Value: RBUnder20000
___|___|___|____Attribute: days.with.cr.line
___|___|___|___|____Value: DaysUnder15Y
___|___|___|___|____Attribute: inq.last.6mths
___|___|___|___|___|____Value: 0
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: 1
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: 2
___|___|___|___|___|____Attribute: log.annual.inc
___|___|___|___|___|___|____Value: LogHigh
___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|___|____Value: LogNormal
___|___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: 3
___|___|___|___|___|____Attribute: log.annual.inc
___|___|___|___|___|___|____Value: LogHigh
___|___|___|___|___|___|____Attribute: delinq.2yrs
___|___|___|___|___|___|___|____Value: 0
___|___|___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|___|___|____Value: 1
___|___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|___|____Value: LogNormal
___|___|___|___|___|___|____Label: PAID
___|___|___|___|____Value: DaysUnder10Y
___|___|___|___|____Attribute: delinq.2yrs
___|___|___|___|___|____Value: 0
___|___|___|___|___|____Attribute: inq.last.6mths
___|___|___|___|___|___|____Value: 0
___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|___|____Value: 1
___|___|___|___|___|___|____Attribute: installment
___|___|___|___|___|___|___|____Value: B200A400
___|___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|___|___|____Value: Under200
___|___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|___|___|____Value: B400A600
___|___|___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|___|____Value: 2
___|___|___|___|___|___|____Attribute: installment
___|___|___|___|___|___|___|____Value: B200A400
___|___|___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|___|___|____Value: Under200
___|___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|___|___|____Value: B400A600
___|___|___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|___|____Value: 3
___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: 1
___|___|___|___|___|____Attribute: installment
___|___|___|___|___|___|____Value: Under200
___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|___|____Value: B400A600
___|___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: 2
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: DaysOver20Y
___|___|___|___|____Attribute: dti
___|___|___|___|___|____Value: DtiGood
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: DtiHigh
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: SoHigh
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: DtiNormal
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: DaysUnder5Y
___|___|___|___|____Attribute: revol.util
___|___|___|___|___|____Value: RVB600A800
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: RVB200A400
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: RVO800
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: RVB400A600
___|___|___|___|___|____Label: PAID
___|___|___|___|____Value: DaysUnder20Y
___|___|___|___|____Attribute: installment
___|___|___|___|___|____Value: B600A800
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: Under200
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: B200A400
___|___|___|___|___|____Attribute: log.annual.inc
___|___|___|___|___|___|____Value: LogHigh
___|___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|___|____Value: LogNormal
___|___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: B400A600
___|___|___|___|___|____Label: NOTPAID
___|___|___|____Value: RBUnder600A80000
___|___|___|____Attribute: installment
___|___|___|___|____Value: B200A400
___|___|___|___|____Label: PAID
___|___|___|___|____Value: O800
___|___|___|___|____Label: NOTPAID
___|___|____Value: credit_card
___|___|____Attribute: dti
___|___|___|____Value: DtiGood
___|___|___|____Label: PAID
___|___|___|____Value: DtiHigh
___|___|___|____Label: PAID
___|___|___|____Value: DtiOptimum
___|___|___|____Label: PAID
___|___|___|____Value: SoHigh
___|___|___|____Label: PAID
___|___|___|____Value: DtiNormal
___|___|___|____Label: NOTPAID
___|___|____Value: educational
___|___|____Attribute: installment
___|___|___|____Value: B200A400
___|___|___|____Label: PAID
___|___|___|____Value: Under200
___|___|___|____Label: NOTPAID
___|___|____Value: small_business
___|___|____Attribute: days.with.cr.line
___|___|___|____Value: DaysUnder15Y
___|___|___|____Attribute: installment
___|___|___|___|____Value: B600A800
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: Under200
___|___|___|___|____Label: PAID
___|___|___|___|____Value: B200A400
___|___|___|___|____Label: PAID
___|___|___|____Value: DaysUnder10Y
___|___|___|____Label: NOTPAID
___|___|___|____Value: DaysOver20Y
___|___|___|____Attribute: inq.last.6mths
___|___|___|___|____Value: 0
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: 1
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: 3
___|___|___|___|____Label: PAID
___|___|___|____Value: DaysUnder20Y
___|___|___|____Label: PAID
___|___|___|____Value: DaysUnder5Y
___|___|___|____Label: NOTPAID
___|___|____Value: major_purchase
___|___|____Attribute: days.with.cr.line
___|___|___|____Value: DaysUnder15Y
___|___|___|____Label: NOTPAID
___|___|___|____Value: DaysOver20Y
___|___|___|____Label: PAID
___|___|___|____Value: DaysUnder20Y
___|___|___|____Label: PAID
___|___|____Value: home_improvement
___|___|____Attribute: days.with.cr.line
___|___|___|____Value: DaysUnder15Y
___|___|___|____Label: PAID
___|___|___|____Value: DaysUnder10Y
___|___|___|____Label: NOTPAID
___|___|___|____Value: DaysUnder20Y
___|___|___|____Attribute: inq.last.6mths
___|___|___|___|____Value: 1
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: 3
___|___|___|___|____Label: PAID
___|___|____Value: all_other
___|___|____Attribute: delinq.2yrs
___|___|___|____Value: 0
___|___|___|____Attribute: days.with.cr.line
___|___|___|___|____Value: DaysUnder15Y
___|___|___|___|____Label: PAID
___|___|___|___|____Value: DaysUnder10Y
___|___|___|___|____Attribute: revol.util
___|___|___|___|___|____Value: RVB600A800
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: RVO800
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: RVUnder200
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: RVB400A600
___|___|___|___|___|____Label: PAID
___|___|___|___|____Value: DaysOver20Y
___|___|___|___|____Label: PAID
___|___|___|___|____Value: DaysUnder5Y
___|___|___|___|____Label: PAID
___|___|___|___|____Value: DaysUnder20Y
___|___|___|___|____Attribute: installment
___|___|___|___|___|____Value: Under200
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: B400A600
___|___|___|___|___|____Label: PAID
___|___|___|____Value: 1
___|___|___|____Label: PAID
___|___|___|____Value: 3
___|___|___|____Label: NOTPAID
___|____Value: Good
___|____Attribute: purpose
___|___|____Value: debt_consolidation
___|___|____Attribute: days.with.cr.line
___|___|___|____Value: DaysUnder15Y
___|___|___|____Label: PAID
___|___|___|____Value: DaysUnder10Y
___|___|___|____Attribute: dti
___|___|___|___|____Value: DtiGood
___|___|___|___|____Label: PAID
___|___|___|___|____Value: DtiOptimum
___|___|___|___|____Label: NOTPAID
___|___|___|____Value: DaysOver20Y
___|___|___|____Label: PAID
___|___|___|____Value: DaysUnder20Y
___|___|___|____Label: PAID
___|___|____Value: credit_card
___|___|____Label: PAID
___|___|____Value: educational
___|___|____Label: PAID
___|___|____Value: small_business
___|___|____Label: PAID
___|___|____Value: major_purchase
___|___|____Attribute: installment
___|___|___|____Value: B200A400
___|___|___|____Label: NOTPAID
___|___|___|____Value: Under200
___|___|___|____Label: PAID
___|___|____Value: home_improvement
___|___|____Label: PAID
___|___|____Value: all_other
___|___|____Attribute: revol.util
___|___|___|____Value: RVB200A400
___|___|___|____Attribute: log.annual.inc
___|___|___|___|____Value: LogHigh
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: LogNormal
___|___|___|___|____Label: PAID
___|___|___|____Value: RVUnder200
___|___|___|____Label: PAID
___|___|___|____Value: RVB400A600
___|___|___|____Label: NOTPAID
___|____Value: Normal
___|____Attribute: inq.last.6mths
___|___|____Value: 0
___|___|____Attribute: purpose
___|___|___|____Value: debt_consolidation
___|___|___|____Label: PAID
___|___|___|____Value: credit_card
___|___|___|____Label: PAID
___|___|___|____Value: educational
___|___|___|____Label: PAID
___|___|___|____Value: small_business
___|___|___|____Attribute: log.annual.inc
___|___|___|___|____Value: LogHigh
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: LogNormal
___|___|___|___|____Label: PAID
___|___|___|____Value: major_purchase
___|___|___|____Label: PAID
___|___|___|____Value: home_improvement
___|___|___|____Label: PAID
___|___|___|____Value: all_other
___|___|___|____Attribute: pub.rec
___|___|___|___|____Value: 0
___|___|___|___|____Label: PAID
___|___|___|___|____Value: 1
___|___|___|___|____Attribute: dti
___|___|___|___|___|____Value: DtiGood
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: DtiHigh
___|___|___|___|___|____Label: NOTPAID
___|___|____Value: 1
___|___|____Attribute: purpose
___|___|___|____Value: debt_consolidation
___|___|___|____Attribute: installment
___|___|___|___|____Value: B200A400
___|___|___|___|____Label: PAID
___|___|___|___|____Value: Under200
___|___|___|___|____Label: PAID
___|___|___|___|____Value: B400A600
___|___|___|___|____Label: PAID
___|___|___|___|____Value: O800
___|___|___|___|____Label: NOTPAID
___|___|___|____Value: credit_card
___|___|___|____Attribute: dti
___|___|___|___|____Value: DtiGood
___|___|___|___|____Label: PAID
___|___|___|___|____Value: DtiHigh
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: SoHigh
___|___|___|___|____Label: PAID
___|___|___|___|____Value: DtiNormal
___|___|___|___|____Label: PAID
___|___|___|____Value: educational
___|___|___|____Label: PAID
___|___|___|____Value: small_business
___|___|___|____Label: PAID
___|___|___|____Value: major_purchase
___|___|___|____Label: PAID
___|___|___|____Value: home_improvement
___|___|___|____Label: PAID
___|___|___|____Value: all_other
___|___|___|____Attribute: revol.util
___|___|___|___|____Value: RVB600A800
___|___|___|___|____Attribute: log.annual.inc
___|___|___|___|___|____Value: LogHigh
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: LogNormal
___|___|___|___|___|____Label: PAID
___|___|___|___|____Value: RVO800
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: RVUnder200
___|___|___|___|____Label: PAID
___|___|___|___|____Value: RVB200A400
___|___|___|___|____Attribute: days.with.cr.line
___|___|___|___|___|____Value: DaysUnder15Y
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: DaysUnder10Y
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: DaysOver20Y
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: DaysUnder5Y
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|___|____Value: DaysUnder20Y
___|___|___|___|___|____Label: PAID
___|___|___|___|____Value: RVB400A600
___|___|___|___|____Label: PAID
___|___|____Value: 2
___|___|____Attribute: purpose
___|___|___|____Value: debt_consolidation
___|___|___|____Label: PAID
___|___|___|____Value: credit_card
___|___|___|____Attribute: installment
___|___|___|___|____Value: B200A400
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: Under200
___|___|___|___|____Label: PAID
___|___|___|____Value: educational
___|___|___|____Label: PAID
___|___|___|____Value: small_business
___|___|___|____Label: PAID
___|___|___|____Value: major_purchase
___|___|___|____Label: PAID
___|___|___|____Value: home_improvement
___|___|___|____Label: NOTPAID
___|___|___|____Value: all_other
___|___|___|____Attribute: revol.util
___|___|___|___|____Value: RVB600A800
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: RVO800
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: RVUnder200
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: RVB200A400
___|___|___|___|____Label: PAID
___|___|___|___|____Value: RVB400A600
___|___|___|___|____Label: PAID
___|___|____Value: 3
___|___|____Attribute: dti
___|___|___|____Value: DtiGood
___|___|___|____Label: PAID
___|___|___|____Value: DtiHigh
___|___|___|____Label: PAID
___|___|___|____Value: DtiOptimum
___|___|___|____Label: PAID
___|___|___|____Value: SoHigh
___|___|___|____Attribute: purpose
___|___|___|___|____Value: debt_consolidation
___|___|___|___|____Attribute: installment
___|___|___|___|___|____Value: B200A400
___|___|___|___|___|____Label: PAID
___|___|___|___|___|____Value: B400A600
___|___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: educational
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: home_improvement
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: all_other
___|___|___|___|____Label: PAID
___|___|___|____Value: DtiNormal
___|___|___|____Attribute: purpose
___|___|___|___|____Value: home_improvement
___|___|___|___|____Label: PAID
___|___|___|___|____Value: all_other
___|___|___|___|____Label: NOTPAID
___|___|____Value: 4
___|___|____Attribute: purpose
___|___|___|____Value: debt_consolidation
___|___|___|____Attribute: installment
___|___|___|___|____Value: B200A400
___|___|___|___|____Label: NOTPAID
___|___|___|___|____Value: O800
___|___|___|___|____Label: PAID
___|___|___|____Value: small_business
___|___|___|____Label: NOTPAID
___|___|___|____Value: home_improvement
___|___|___|____Label: PAID
___|___|___|____Value: all_other
___|___|___|____Label: PAID
___|___|____Value: 5
___|___|____Attribute: purpose
___|___|___|____Value: small_business
___|___|___|____Label: NOTPAID
___|___|___|____Value: all_other
___|___|___|____Label: PAID
___|___|____Value: 6
___|___|____Label: PAID
___|____Value: SoHigh
___|____Attribute: purpose
___|___|____Value: debt_consolidation
___|___|____Attribute: dti
___|___|___|____Value: DtiGood
___|___|___|____Label: NOTPAID
___|___|___|____Value: DtiHigh
___|___|___|____Label: PAID
___|___|___|____Value: DtiOptimum
___|___|___|____Label: NOTPAID
___|___|___|____Value: SoHigh
___|___|___|____Label: PAID
___|___|____Value: credit_card
___|___|____Label: PAID
___|___|____Value: small_business
___|___|____Attribute: installment
___|___|___|____Value: B600A800
___|___|___|____Label: PAID
___|___|___|____Value: Under200
___|___|___|____Attribute: dti
___|___|___|___|____Value: DtiOptimum
___|___|___|___|____Label: PAID
___|___|___|___|____Value: SoHigh
___|___|___|___|____Label: NOTPAID
___|___|___|____Value: B200A400
___|___|___|____Label: NOTPAID
___|___|___|____Value: B400A600
___|___|___|____Label: PAID
___|___|___|____Value: O800
___|___|___|____Label: NOTPAID
___|___|____Value: major_purchase
___|___|____Label: PAID
___|___|____Value: home_improvement
___|___|____Label: NOTPAID
___|___|____Value: all_other
___|___|____Label: PAID


