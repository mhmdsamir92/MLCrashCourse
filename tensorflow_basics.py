import tensorflow as tf

# TensorFlow gets its name from tensors, which are arrays of arbitrary dimensionality.
# Using TensorFlow, you can manipulate tensors with a very high number of dimensions.
# That said, most of the time you will work with one or more of the following low-dimensional
# tensors:
#   A scalar is a 0-d array (a 0th-order tensor). For example, "Howdy" or 5
#   A vector is a 1-d array (a 1st-order tensor). For example, [2, 3, 5, 7, 11] or [5]
#   A matrix is a 2-d array (a 2nd-order tensor). For example, [[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]

# A TensorFlow graph (also known as a computational graph or a dataflow graph) is, yes,
# a graph data structure. A graph's nodes are operations (in TensorFlow, every operation
# is associated with a graph). Many TensorFlow programs consist of a single graph,
# but TensorFlow programs may optionally create multiple graphs.
# A graph's nodes are operations; a graph's edges are tensors.
# Tensors flow through the graph, manipulated at each node by an operation.
# The output tensor of one operation often becomes the input tensor to a subsequent operation.
# TensorFlow implements a lazy execution model, meaning that nodes are only computed when
# needed, based on the needs of associated nodes.

# Tensors can be stored in the graph as constants or variables. As you might guess,
# constants hold tensors whose values can't change, while variables hold tensors whose values can change.
# However, what you may not have guessed is that constants and variables are just more operations in the
# graph. A constant is an operation that always returns the same tensor value. 
# A variable is an operation that will return whichever tensor has been assigned to it.

x = tf.constant(5.2)
y = tf.Variable([5])

y = tf.Variable([0])
y = y.assign([5])

with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    print y.eval()

# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
    # Assemble a graph consisting of the following three operations:
    #   * Two tf.constant operations to create the operands.
    #   * One tf.add operation to add the two operands.
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    my_sum = tf.add(x, y, name="x_y_sum")
    # Now create a session.
    # The session will run the default graph.
    with tf.Session() as sess:
        print my_sum.eval()

# Create a graph.
g = tf.Graph()

# Establish our graph as the "default" graph.
with g.as_default():
    # Assemble a graph consisting of three operations.
    # (Creating a tensor is an operation.)
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    my_sum = tf.add(x, y, name="x_y_sum")

    # Task 1: Define a third scalar integer constant z.
    z = tf.constant(4, name="z_const")
    # Task 2: Add z to `my_sum` to yield a new sum.
    new_sum = tf.add(my_sum, z, name="x_y_z_sum")

    # Now create a session.
    # The session will run the default graph.
    with tf.Session() as sess:
        # Task 3: Ensure the program yields the correct grand total.
        print new_sum.eval()
