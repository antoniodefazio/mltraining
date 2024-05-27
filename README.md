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

----Attribute: int.rate
---|----Value: High
---|----Attribute: purpose
---|---|----Value: debt_consolidation
---|---|----Attribute: revol.bal
---|---|---|----Value: RBUnder400A60000
---|---|---|----Label: PAID
---|---|---|----Value: RBUnder200A40000
---|---|---|----Attribute: installment
---|---|---|---|----Value: B600A800
---|---|---|---|----Label: PAID
---|---|---|---|----Value: Under200
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: B200A400
---|---|---|---|----Attribute: dti
---|---|---|---|---|----Value: DtiGood
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: SoHigh
---|---|---|---|---|----Label: PAID
---|---|---|---|----Value: B400A600
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: O800
---|---|---|---|----Label: PAID
---|---|---|----Value: RBOver80000
---|---|---|----Label: PAID
---|---|---|----Value: RBUnder20000
---|---|---|----Attribute: days.with.cr.line
---|---|---|---|----Value: DaysUnder15Y
---|---|---|---|----Attribute: inq.last.6mths
---|---|---|---|---|----Value: 0
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: 1
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: 2
---|---|---|---|---|----Attribute: log.annual.inc
---|---|---|---|---|---|----Value: LogHigh
---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|---|----Value: LogNormal
---|---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: 3
---|---|---|---|---|----Attribute: log.annual.inc
---|---|---|---|---|---|----Value: LogHigh
---|---|---|---|---|---|----Attribute: delinq.2yrs
---|---|---|---|---|---|---|----Value: 0
---|---|---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|---|---|----Value: 1
---|---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|---|----Value: LogNormal
---|---|---|---|---|---|----Label: PAID
---|---|---|---|----Value: DaysUnder10Y
---|---|---|---|----Attribute: delinq.2yrs
---|---|---|---|---|----Value: 0
---|---|---|---|---|----Attribute: inq.last.6mths
---|---|---|---|---|---|----Value: 0
---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|---|----Value: 1
---|---|---|---|---|---|----Attribute: installment
---|---|---|---|---|---|---|----Value: B200A400
---|---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|---|---|----Value: Under200
---|---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|---|---|----Value: B400A600
---|---|---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|---|----Value: 2
---|---|---|---|---|---|----Attribute: installment
---|---|---|---|---|---|---|----Value: B200A400
---|---|---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|---|---|----Value: Under200
---|---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|---|---|----Value: B400A600
---|---|---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|---|----Value: 3
---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: 1
---|---|---|---|---|----Attribute: installment
---|---|---|---|---|---|----Value: Under200
---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|---|----Value: B400A600
---|---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: 2
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: DaysOver20Y
---|---|---|---|----Attribute: dti
---|---|---|---|---|----Value: DtiGood
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: DtiHigh
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: SoHigh
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: DtiNormal
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: DaysUnder5Y
---|---|---|---|----Attribute: revol.util
---|---|---|---|---|----Value: RVB600A800
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: RVB200A400
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: RVO800
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: RVB400A600
---|---|---|---|---|----Label: PAID
---|---|---|---|----Value: DaysUnder20Y
---|---|---|---|----Attribute: installment
---|---|---|---|---|----Value: B600A800
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: Under200
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: B200A400
---|---|---|---|---|----Attribute: log.annual.inc
---|---|---|---|---|---|----Value: LogHigh
---|---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|---|----Value: LogNormal
---|---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: B400A600
---|---|---|---|---|----Label: NOTPAID
---|---|---|----Value: RBUnder600A80000
---|---|---|----Attribute: installment
---|---|---|---|----Value: B200A400
---|---|---|---|----Label: PAID
---|---|---|---|----Value: O800
---|---|---|---|----Label: NOTPAID
---|---|----Value: credit_card
---|---|----Attribute: dti
---|---|---|----Value: DtiGood
---|---|---|----Label: PAID
---|---|---|----Value: DtiHigh
---|---|---|----Label: PAID
---|---|---|----Value: DtiOptimum
---|---|---|----Label: PAID
---|---|---|----Value: SoHigh
---|---|---|----Label: PAID
---|---|---|----Value: DtiNormal
---|---|---|----Label: NOTPAID
---|---|----Value: educational
---|---|----Attribute: installment
---|---|---|----Value: B200A400
---|---|---|----Label: PAID
---|---|---|----Value: Under200
---|---|---|----Label: NOTPAID
---|---|----Value: small_business
---|---|----Attribute: days.with.cr.line
---|---|---|----Value: DaysUnder15Y
---|---|---|----Attribute: installment
---|---|---|---|----Value: B600A800
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: Under200
---|---|---|---|----Label: PAID
---|---|---|---|----Value: B200A400
---|---|---|---|----Label: PAID
---|---|---|----Value: DaysUnder10Y
---|---|---|----Label: NOTPAID
---|---|---|----Value: DaysOver20Y
---|---|---|----Attribute: inq.last.6mths
---|---|---|---|----Value: 0
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: 1
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: 3
---|---|---|---|----Label: PAID
---|---|---|----Value: DaysUnder20Y
---|---|---|----Label: PAID
---|---|---|----Value: DaysUnder5Y
---|---|---|----Label: NOTPAID
---|---|----Value: major_purchase
---|---|----Attribute: days.with.cr.line
---|---|---|----Value: DaysUnder15Y
---|---|---|----Label: NOTPAID
---|---|---|----Value: DaysOver20Y
---|---|---|----Label: PAID
---|---|---|----Value: DaysUnder20Y
---|---|---|----Label: PAID
---|---|----Value: home_improvement
---|---|----Attribute: days.with.cr.line
---|---|---|----Value: DaysUnder15Y
---|---|---|----Label: PAID
---|---|---|----Value: DaysUnder10Y
---|---|---|----Label: NOTPAID
---|---|---|----Value: DaysUnder20Y
---|---|---|----Attribute: inq.last.6mths
---|---|---|---|----Value: 1
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: 3
---|---|---|---|----Label: PAID
---|---|----Value: all_other
---|---|----Attribute: delinq.2yrs
---|---|---|----Value: 0
---|---|---|----Attribute: days.with.cr.line
---|---|---|---|----Value: DaysUnder15Y
---|---|---|---|----Label: PAID
---|---|---|---|----Value: DaysUnder10Y
---|---|---|---|----Attribute: revol.util
---|---|---|---|---|----Value: RVB600A800
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: RVO800
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: RVUnder200
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: RVB400A600
---|---|---|---|---|----Label: PAID
---|---|---|---|----Value: DaysOver20Y
---|---|---|---|----Label: PAID
---|---|---|---|----Value: DaysUnder5Y
---|---|---|---|----Label: PAID
---|---|---|---|----Value: DaysUnder20Y
---|---|---|---|----Attribute: installment
---|---|---|---|---|----Value: Under200
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: B400A600
---|---|---|---|---|----Label: PAID
---|---|---|----Value: 1
---|---|---|----Label: PAID
---|---|---|----Value: 3
---|---|---|----Label: NOTPAID
---|----Value: Good
---|----Attribute: purpose
---|---|----Value: debt_consolidation
---|---|----Attribute: days.with.cr.line
---|---|---|----Value: DaysUnder15Y
---|---|---|----Label: PAID
---|---|---|----Value: DaysUnder10Y
---|---|---|----Attribute: dti
---|---|---|---|----Value: DtiGood
---|---|---|---|----Label: PAID
---|---|---|---|----Value: DtiOptimum
---|---|---|---|----Label: NOTPAID
---|---|---|----Value: DaysOver20Y
---|---|---|----Label: PAID
---|---|---|----Value: DaysUnder20Y
---|---|---|----Label: PAID
---|---|----Value: credit_card
---|---|----Label: PAID
---|---|----Value: educational
---|---|----Label: PAID
---|---|----Value: small_business
---|---|----Label: PAID
---|---|----Value: major_purchase
---|---|----Attribute: installment
---|---|---|----Value: B200A400
---|---|---|----Label: NOTPAID
---|---|---|----Value: Under200
---|---|---|----Label: PAID
---|---|----Value: home_improvement
---|---|----Label: PAID
---|---|----Value: all_other
---|---|----Attribute: revol.util
---|---|---|----Value: RVB200A400
---|---|---|----Attribute: log.annual.inc
---|---|---|---|----Value: LogHigh
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: LogNormal
---|---|---|---|----Label: PAID
---|---|---|----Value: RVUnder200
---|---|---|----Label: PAID
---|---|---|----Value: RVB400A600
---|---|---|----Label: NOTPAID
---|----Value: Normal
---|----Attribute: inq.last.6mths
---|---|----Value: 0
---|---|----Attribute: purpose
---|---|---|----Value: debt_consolidation
---|---|---|----Label: PAID
---|---|---|----Value: credit_card
---|---|---|----Label: PAID
---|---|---|----Value: educational
---|---|---|----Label: PAID
---|---|---|----Value: small_business
---|---|---|----Attribute: log.annual.inc
---|---|---|---|----Value: LogHigh
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: LogNormal
---|---|---|---|----Label: PAID
---|---|---|----Value: major_purchase
---|---|---|----Label: PAID
---|---|---|----Value: home_improvement
---|---|---|----Label: PAID
---|---|---|----Value: all_other
---|---|---|----Attribute: pub.rec
---|---|---|---|----Value: 0
---|---|---|---|----Label: PAID
---|---|---|---|----Value: 1
---|---|---|---|----Attribute: dti
---|---|---|---|---|----Value: DtiGood
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: DtiHigh
---|---|---|---|---|----Label: NOTPAID
---|---|----Value: 1
---|---|----Attribute: purpose
---|---|---|----Value: debt_consolidation
---|---|---|----Attribute: installment
---|---|---|---|----Value: B200A400
---|---|---|---|----Label: PAID
---|---|---|---|----Value: Under200
---|---|---|---|----Label: PAID
---|---|---|---|----Value: B400A600
---|---|---|---|----Label: PAID
---|---|---|---|----Value: O800
---|---|---|---|----Label: NOTPAID
---|---|---|----Value: credit_card
---|---|---|----Attribute: dti
---|---|---|---|----Value: DtiGood
---|---|---|---|----Label: PAID
---|---|---|---|----Value: DtiHigh
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: SoHigh
---|---|---|---|----Label: PAID
---|---|---|---|----Value: DtiNormal
---|---|---|---|----Label: PAID
---|---|---|----Value: educational
---|---|---|----Label: PAID
---|---|---|----Value: small_business
---|---|---|----Label: PAID
---|---|---|----Value: major_purchase
---|---|---|----Label: PAID
---|---|---|----Value: home_improvement
---|---|---|----Label: PAID
---|---|---|----Value: all_other
---|---|---|----Attribute: revol.util
---|---|---|---|----Value: RVB600A800
---|---|---|---|----Attribute: log.annual.inc
---|---|---|---|---|----Value: LogHigh
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: LogNormal
---|---|---|---|---|----Label: PAID
---|---|---|---|----Value: RVO800
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: RVUnder200
---|---|---|---|----Label: PAID
---|---|---|---|----Value: RVB200A400
---|---|---|---|----Attribute: days.with.cr.line
---|---|---|---|---|----Value: DaysUnder15Y
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: DaysUnder10Y
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: DaysOver20Y
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: DaysUnder5Y
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|---|----Value: DaysUnder20Y
---|---|---|---|---|----Label: PAID
---|---|---|---|----Value: RVB400A600
---|---|---|---|----Label: PAID
---|---|----Value: 2
---|---|----Attribute: purpose
---|---|---|----Value: debt_consolidation
---|---|---|----Label: PAID
---|---|---|----Value: credit_card
---|---|---|----Attribute: installment
---|---|---|---|----Value: B200A400
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: Under200
---|---|---|---|----Label: PAID
---|---|---|----Value: educational
---|---|---|----Label: PAID
---|---|---|----Value: small_business
---|---|---|----Label: PAID
---|---|---|----Value: major_purchase
---|---|---|----Label: PAID
---|---|---|----Value: home_improvement
---|---|---|----Label: NOTPAID
---|---|---|----Value: all_other
---|---|---|----Attribute: revol.util
---|---|---|---|----Value: RVB600A800
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: RVO800
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: RVUnder200
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: RVB200A400
---|---|---|---|----Label: PAID
---|---|---|---|----Value: RVB400A600
---|---|---|---|----Label: PAID
---|---|----Value: 3
---|---|----Attribute: dti
---|---|---|----Value: DtiGood
---|---|---|----Label: PAID
---|---|---|----Value: DtiHigh
---|---|---|----Label: PAID
---|---|---|----Value: DtiOptimum
---|---|---|----Label: PAID
---|---|---|----Value: SoHigh
---|---|---|----Attribute: purpose
---|---|---|---|----Value: debt_consolidation
---|---|---|---|----Attribute: installment
---|---|---|---|---|----Value: B200A400
---|---|---|---|---|----Label: PAID
---|---|---|---|---|----Value: B400A600
---|---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: educational
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: home_improvement
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: all_other
---|---|---|---|----Label: PAID
---|---|---|----Value: DtiNormal
---|---|---|----Attribute: purpose
---|---|---|---|----Value: home_improvement
---|---|---|---|----Label: PAID
---|---|---|---|----Value: all_other
---|---|---|---|----Label: NOTPAID
---|---|----Value: 4
---|---|----Attribute: purpose
---|---|---|----Value: debt_consolidation
---|---|---|----Attribute: installment
---|---|---|---|----Value: B200A400
---|---|---|---|----Label: NOTPAID
---|---|---|---|----Value: O800
---|---|---|---|----Label: PAID
---|---|---|----Value: small_business
---|---|---|----Label: NOTPAID
---|---|---|----Value: home_improvement
---|---|---|----Label: PAID
---|---|---|----Value: all_other
---|---|---|----Label: PAID
---|---|----Value: 5
---|---|----Attribute: purpose
---|---|---|----Value: small_business
---|---|---|----Label: NOTPAID
---|---|---|----Value: all_other
---|---|---|----Label: PAID
---|---|----Value: 6
---|---|----Label: PAID
---|----Value: SoHigh
---|----Attribute: purpose
---|---|----Value: debt_consolidation
---|---|----Attribute: dti
---|---|---|----Value: DtiGood
---|---|---|----Label: NOTPAID
---|---|---|----Value: DtiHigh
---|---|---|----Label: PAID
---|---|---|----Value: DtiOptimum
---|---|---|----Label: NOTPAID
---|---|---|----Value: SoHigh
---|---|---|----Label: PAID
---|---|----Value: credit_card
---|---|----Label: PAID
---|---|----Value: small_business
---|---|----Attribute: installment
---|---|---|----Value: B600A800
---|---|---|----Label: PAID
---|---|---|----Value: Under200
---|---|---|----Attribute: dti
---|---|---|---|----Value: DtiOptimum
---|---|---|---|----Label: PAID
---|---|---|---|----Value: SoHigh
---|---|---|---|----Label: NOTPAID
---|---|---|----Value: B200A400
---|---|---|----Label: NOTPAID
---|---|---|----Value: B400A600
---|---|---|----Label: PAID
---|---|---|----Value: O800
---|---|---|----Label: NOTPAID
---|---|----Value: major_purchase
---|---|----Label: PAID
---|---|----Value: home_improvement
---|---|----Label: NOTPAID
---|---|----Value: all_other
---|---|----Label: PAID
