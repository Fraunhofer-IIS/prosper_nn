Fuzzy Neural Network
--------------------

Member Layer
------------

Modules
^^^^^^^

Different membership functions are defined, which can be used in member_block.

.. automodule:: prosper_nn.models.fuzzy.member_functions
    :members:
    :undoc-members:
    :show-inheritance:

The member_block class uses the functions defined in member_functions and applies them to a single input.

.. automodule:: prosper_nn.models.fuzzy.member_block
    :members:
    :undoc-members:
    :show-inheritance:

The final member_layer results by setting a member_block for each input of the layer.

.. automodule:: prosper_nn.models.fuzzy.member_layer
    :members:
    :undoc-members:
    :show-inheritance:
	
Example
^^^^^^^

.. code:: python

	batchsize = 20
	n_features_inputs = 10

	inputs = torch.randn(batchsize, n_features_input)

	

	members = {"neg_func": NormlogMember(negative=True),
               "const_func": GaussianMember(),
               "pos_func": NormlogMember()}
			   
	member_layer = MemberLayer(n_features_input=n_features_input,
                               member_functions=members)
			   
	output = member_layer(inputs)

    
Rule Dense Layer
----------------

Module
^^^^^^

The RuleDenseLayer gets the output of the MemberLayer. It is a Dense Layer that has a predefined weights matrix `rule_matrix`.
Use the RuleManager to create the rule matrix.

.. automodule:: prosper_nn.models.fuzzy.rule_dense
    :members:
    :undoc-members:
    :show-inheritance:
	
Example
^^^^^^^
.. code:: python

	batchsize = 20
	n_features_inputs = 10
	n_members = 3
	n_features_output = 5

	inputs = torch.randn(batchsize, n_features_input, n_members)
	
	dummy_rule_matrix = np.ones(n_features_output, n_features_input, n_members)
	
	rule_dense = RuleDenseLayer(n_features_input=n_features_input,
                                n_features_output=n_features_output,
                                n_member_functions=n_members,
                                rule_matrix=dummy_rule_matrix,
                                learn=False,
                                prune_weights=True)

	output = rule_dense(inputs)
	
Defuzzification Layer
---------------------

Module
^^^^^^

The DefuzzificationLayer turns the rule output into a prediction by applying the classification_matrix to the input.
Use the RuleManager module to create the classification_matrix from  JSON.

.. automodule:: prosper_nn.models.fuzzy.defuzzification_layer
    :members:
    :undoc-members:
    :show-inheritance:
	
Example
^^^^^^^

.. code:: python

	n_features_input = 5
	n_features_output = 3
	batchsize = 20

	
	dummy_classification_matrix = np.ones(n_features_input, n_features_output)
	
	inputs = torch.randn(batchsize, n_features_input)
	
	defuzzification_layer = DefuzzificationLayer(
        n_features_input=n_features_input,
        n_features_output=n_features_output,
        mode="classification",
        classification_matrix=dummy_classification_matrix,
        learn=True))
								 
	output = defuzzification_layer(input)
	
	
Fuzzy Recurrent Neural Networks
-------------------------------

Module
^^^^^^^

The Fuzzy Recurrent Neural Network (FRNN) uses a special RNN infront of a Fuzzy Neural Net.
The RNN is used to calculate the inputs' change over time. The RNN is pruned to not mix different inputs. The FNN analyses the inputs' changes afterwards.
The member, rule_dense and defuzzification layers from the prosper package are used to build a FRNN Neural Network.

.. automodule:: prosper_nn.models.fuzzy.frnn
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

	batchsize = 30
	sequence_length = 20
	n_features_input = 10
	n_features_output = 3
	n_rules = 5
	n_members = 3	
	
	inputs = torch.randn(batchsize, sequence_length, n_features_input)
	
	
	dummy_rule_matrix = np.ones(n_rules, n_features_input, n_members)
	dummy_classification_matrix = np.ones(n_rules, n_features_output)
	
	members = {"neg_func": NormlogMember(negative=True),
               "const_func": GaussianMember(),
               "pos_func": NormlogMember()}
			   
	frnn = FRNN(n_features_input=n_features_input,
				n_features_output=n_features_output,
				n_rules=n_rules,
				n_members=n_members,
				members=members,
                rule_matrix=dummy_rule_matrix,
                classification_matrix=dummy_classification_matrix,
                mode = "classification",
                learn: bool = False,
                pruning: bool = True)
				
	output = frnn(inputs)
