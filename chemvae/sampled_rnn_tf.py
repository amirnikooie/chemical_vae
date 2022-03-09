'''
sampled_rnn - based on rnn method in tensorflow_backend.py in keras

Main difference is in how to handle dimensions of states.


# think carefully about the distribution of the random sampled variables...

'''
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops, control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
import tensorflow.keras.backend as K
from tensorflow.python.util import nest

def sampled_rnn(step_function, inputs, initial_states, units, random_seed,
                go_backwards=False, mask=None, rec_dp_constants=None,
                unroll=False, input_length=None, constants=None, time_major = False):
    """Iterates over the time dimension of a tensor.
    # Arguments
        step_function: RNN step function.
            Parameters:
                input: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        units : number of units in the output dimension.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

            last_output: the latest output of the rnn, of shape `(samples, ...)`
            outputs: tensor with shape `(samples, time, ...)` where each
                etry `outputs[s, t]` is the output of the step function
                at time `t` for sample `s`.
            new_states: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: if input dimension is less than 3.
        ValueError: if `unroll` is `True` but input timestep is not a fixed number.
        ValueError: if `mask` is provided (not `None`) but states is not provided
            (`len(states)` == 0).
   """
    np.random.seed(random_seed) # This line and the next line are unique to this code. So, this is a difference.
    tf.random.set_seed(random_seed)
    ndim = len(inputs.get_shape())
    if ndim < 3:
        raise ValueError('Input should be at least 3D.')

    if unroll == True:
        raise ValueError('Unrolling not implemented in sampled_rnn')
    if mask is not None:
        raise ValueError('Masking not implemented in sampled_rnn')

    def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return array_ops.transpose(input_t, axes)

    if not time_major:
        inputs = nest.map_structure(swap_batch_timestep, inputs)

    flatted_inputs = nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = array_ops.shape(flatted_inputs[0])[0]

    for input_ in flatted_inputs:
        input_.shape.with_rank_at_least(3)

    if constants is None:
        constants = []

    states = tuple(initial_states)

    #=== Customized block start ======
    if go_backwards: # This will not be exectued actually due to go_backwards=False
        inputs = K.reverse(inputs, 0)

    num_samples = tf.shape(inputs)[1]
    output_dim = int(initial_states[0].get_shape()[-1])
    random_cutoff_prob = tf.random.uniform(
        (num_samples,), minval=0., maxval=1.)
    #=== Customized block end ======

    # Create input tensor array, if the inputs is nested tensors, then it will
    # be flattened first, and tensor array will be created one per flattened
    # tensor.
    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    input_ta = tuple(
        ta.unstack(input_) if not go_backwards else ta
        .unstack(K.reverse(input_, 0))
        for ta, input_ in zip(input_ta, flatted_inputs))

    # Get the time(0) input and compute the output for that, the output will be
    # used to determine the dtype of output tensor array. Don't read from
    # input_ta due to TensorArray clear_after_read default to True.
    input_time_zero = nest.pack_sequence_as(inputs,
                                            [inp[0] for inp in flatted_inputs])
    # output_time_zero is used to determine the cell output shape and its dtype.
    # the value is discarded.

    #===== Customized input arguments for step_function start ======
    output_time_zero, _ = step_function(inputs[0], {'initial_states': initial_states, # Different input arguments to this function compared to the original code. so this a difference.
                                           'random_cutoff_prob': random_cutoff_prob,
                                           'rec_dp_mask': rec_dp_constants})
    #===== Customized input arguments for step_function end ======

    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            element_shape=out.shape,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))

    time = constant_op.constant(0, dtype='int32', name='time')

    # We only specify the 'maximum_iterations' when building for XLA since that
    # causes slowdowns on GPU in TF.
    if (not context.executing_eagerly() and
        control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())):
      max_iterations = math_ops.reduce_max(input_length)
    else:
      max_iterations = None

    while_loop_kwargs = {
        'cond': lambda time, *_: time < time_steps_t,
        'maximum_iterations': max_iterations,
        'parallel_iterations': 1,
        'swap_memory': True,
    }

    masking_fn = None # Because of the settings for this code.

    def _step(time, output_ta_t, *states):
        """RNN step function.
        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.
        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        #==== Customized section start ====
        random_cutoff_prob = tf.random.uniform(
            (num_samples,), minval=0, maxval=1)
        #==== Customized section end ====

    #===== Customized input arguments for step_function start ======
        output, new_states = step_function(current_input,
                                           {'initial_states': states,
                                            'random_cutoff_prob': random_cutoff_prob,
                                            'rec_dp_mask': rec_dp_constants})
    #===== Customized input arguments for step_function end ======

        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)

        #==== Customized section start ====
        axes = [1, 0] + list(range(2, K.ndim(output)))
        output = tf.transpose(output, (axes))
        #==== Customized section end ====

#        for state, new_state in zip(flat_state, flat_new_state):
#          if isinstance(new_state, ops.Tensor):
#            new_state.set_shape(state.shape)

        flat_output = nest.flatten(output)
        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
        new_states = nest.pack_sequence_as(initial_states, flat_new_state)
        return (time + 1, output_ta_t) + tuple(new_states)

    final_outputs = control_flow_ops.while_loop(
        body=_step,
        loop_vars=(time, output_ta) + states,
        **while_loop_kwargs)

    new_states = final_outputs[2:]
    output_ta = final_outputs[1]
    outputs = tuple(o.stack() for o in output_ta)
    last_output = tuple(o[-1] for o in outputs)

    outputs = nest.pack_sequence_as(output_time_zero, outputs)
    last_output = nest.pack_sequence_as(output_time_zero, last_output)

  # static shape inference
    def set_shape(output_):
        if isinstance(output_, ops.Tensor):
            shape = output_.shape.as_list()
            shape[0] = time_steps
            shape[1] = batch
            output_.set_shape(shape)
    return output_

    outputs = nest.map_structure(set_shape, outputs)

    #==== Customized section start ====
    outputs = output_ta.stack()[:, :, 1, :] # This brackets in this line and the next line is one difference.
    last_output = output_ta.read(last_time - 1)[:, 1, :]
    #==== Customized section start ====

    if not time_major:
        outputs = nest.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states
