Fuzzy Neural Network
--------------------

Fuzzification
--------------

Modules
^^^^^^^

Different membership functions are defined, which can be used in membership_block.

.. automodule:: prosper_nn.models.fuzzy.membership_functions
    :members:
    :undoc-members:
    :show-inheritance:

The membership_block class uses the functions defined in membership_functions and applies them to a single input.

.. automodule:: prosper_nn.models.fuzzy.membership_block
    :members:
    :undoc-members:
    :show-inheritance:

The final member_layer results by setting a membership_block for each input of the layer.

.. automodule:: prosper_nn.models.fuzzy.fuzzification
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

	batchsize = 20
	n_features_inputs = 11

	inputs = torch.randn(batchsize, n_features_input)



    membership_fcts = {
        "decrease": NormlogMembership(negative=True),
        "constant": GaussianMembership(),
        "increase": NormlogMembership(),
    }

	fuzzification = Fuzzification(
        n_features_input=n_features_input, membership_fcts=membership_fcts
    )

	output = fuzzification(inputs)


Fuzzy Inference
----------------

Module
^^^^^^

The FuzzyInference gets the output of the Fuzzification. It is a Dense Layer that has a predefined weights matrix `rule_matrix`.
Use the RuleManager to create the rule matrix and classification matrix.

.. automodule:: prosper_nn.models.fuzzy.fuzzy_inference
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^
.. code:: python

    batchsize = 20
	n_features_input = 11
    n_output_classes = 3
    n_rules = 4
    n_membership_fcts = 3

	inputs = torch.randn(batchsize, n_features_input, n_membership_fcts)

	dummy_rule_matrix = np.ones(n_features_output, n_features_input, n_members)
    dummy_classification_matrix = np.ones(n_features_input, n_features_output)

    fuzzy_inference = FuzzyInference(
        n_features_input=n_features_input,
        n_rules=n_rules,
        n_output_classes=n_output_classes,
        n_membership_fcts=n_membership_fcts,
        rule_matrix=dummy_rule_matrix,
        classification_matrix=dummy_classification_matrix,
    )

	output = fuzzy_inference(inputs)

Defuzzification
---------------------

Module
^^^^^^

The Defuzzification turns the interpretable class prediction into a numerical prediction.

.. automodule:: prosper_nn.models.fuzzy.defuzzification
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

	n_output_classes = 3
	batchsize = 20

	inputs = torch.randn(batchsize, n_output_classes)

	defuzzification = Defuzzification(n_output_classes)

	output = defuzzification(input)


Fuzzy Recurrent Neural Networks
-------------------------------

Module
^^^^^^^

The Fuzzy Recurrent Neural Network (FRNN) uses a special RNN infront of a Fuzzy Neural Net.
The RNN is used to calculate the inputs' change over time. The RNN is pruned to not mix different inputs. The FNN analyses the inputs' changes afterwards.
The fuzzification, fuzzy inference and defuzzification layers are used to build a FRNN Neural Network.

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
	n_output_classes = 3
	n_rules = 5
	n_membership_fcts = 3

	inputs = torch.randn(batchsize, sequence_length, n_features_input)

	dummy_rule_matrix = np.ones(n_rules, n_features_input, n_membership_fcts)
	dummy_classification_matrix = np.ones(n_rules, n_output_classes)

	membership_fcts = {
        "negative": NormlogMembership(negative=True),
        "constant": GaussianMembership(),
        "positive": NormlogMembership(),
    }

	frnn = FRNN(
        n_features_input=n_features_input,
		n_output_classes=n_output_classes,
		n_rules=n_rules,
		n_membership_fcts=n_membership_fcts,
		membership_fcts=membership_fcts,
        rule_matrix=dummy_rule_matrix,
        classification_matrix=dummy_classification_matrix,
        )

	output = frnn(inputs)
