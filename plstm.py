# 
#          定义 phased lstm 层 
#



import numpy as np
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper

from lasagne.layers.recurrent import Gate

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng


class PLSTMTimeGate(object):
    """
        返回一个 时间门的类 ， 包含属性 周期， 相位，开放的时间长度比例

    
    """
    def __init__(self,
                 Period=init.Uniform((10,100)),  # 初始化周期
                 Shift=init.Uniform( (0., 1000.)),  # 初始化相位
                 On_End=init.Constant(0.05)):   #  开放时间长度比例
        
        self.Period = Period
        self.Shift = Shift
        self.On_End = On_End

class PLSTMLayer(MergeLayer):
    r"""

        返回一个PLSTM 层 类

        输入属性有  incoming , incoming 包含了 x 的输入
        
    """
    def __init__(self,

                 
                 incoming,    #  就是x的输入  (None, features)
                 time_input,  #  每一个 batch的时间点 信息  ,int
                 mask_input=None,  #  time_step 决定哪些
                 cell_init=init.Constant(0.),  #  细胞状态初始化
                 hid_init=init.Constant(0.),  # 隐含层状态初始化

                 num_units,  # 神经元节点数

                 
                 
                 ingate=Gate(b=lasagne.init.Constant(0)),  # 初始化输入门
                 
                 forgetgate=Gate(b=lasagne.init.Constant(2),nonlinearity=nonlinearities.sigmoid), # 初始化遗忘门

                 timegate=PLSTMTimeGate(),  # 创建一个时间门

                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),  # 创建细胞状态控制的门
                 
                 outgate=Gate(), # 初始化输出门

                 
                 nonlinearity=nonlinearities.tanh,  # 这一层的输出的 非线性激活函数

                 backwards=False,

                 learn_init=False,
                 
                 peepholes=True,   #  是否利用窥视孔连接

                 gradient_steps=-1,

                 
                 grad_clipping=0,

                 
                 unroll_scan=False,
                 
                 precompute_input=True,
                 
                 only_return_final=False,
                 
                 bn=False,  # 是否采用另外一种 结构  BN-LSTM

                 learn_time_params=[True, True, False],  # 是否学习 事件门的参数  

                 off_alpha=1e-3,   #  leak rate
                 
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        
        incomings = [incoming]

        
        # TIME STUFF
        
        incomings.append(time_input)
        
        self.time_incoming_index = len(incomings)-1

        self.mask_incoming_index = -2
        self.hid_init_incoming_index = -2
        self.cell_init_incoming_index = -2



        #  incomings 是总的输入  原则上包含了
        #  x 的输入  ，  mask 输入 ,  初始化的隐含层状态输入，  初始化的 细胞状态输入

        #  上述 的 赋值 是给了每一种参数  在 incoming 列表里面的 索引信息


        # 一开始的 incoming 不用必须包含  mask , init_b,  init_c
        # 下面如果有输入的话再加上去  incomings.append()
        
        #ADD TIME INPUT HERE
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1  # 更新索引信息
            
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1 # 更新索引信息
            
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1 # 更新索引信息

        

        # Initialize parent layer
        super(PLSTMLayer, self).__init__(incomings, **kwargs)



        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        #  这段还不知道是干嘛的
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")


        # Retrieve the dimensionality of the incoming layer
        # 检索传入图层的维度
        
        input_shape = self.input_shapes[0]
        
        time_shape = self.input_shapes[1]

        ## unroll_scan 展开扫描
        # 如果要展开扫描  ，  那么时间 维度的 shape 就不能为None
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        #  num_inputs == features np.prod 是吧除了 batch_size-- input_shape[0] , time_step--input_shape[1]以外的其余
        #  维度数全部相乘，当成features
        num_inputs = np.prod(input_shape[2:])


        def add_gate_params(gate, gate_name):  #  指定门的 变量，  和名字(名字后面有用)
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            # 用于从Gate实例添加图层参数的便捷功能
            return (self.add_param(gate.W_in, (num_inputs, num_units), # 这里相当于给的 shape ，用initializer 类 进行初始化
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                gate.nonlinearity)


        # PHASED LSTM: Initialize params for the time gate
        #  初始化时间门的 参数 

        self.off_alpha = off_alpha
        # leak_rate
        
        if timegate == None: #  如果时间门为空的话， 那么指定生成一个 时间门 instance 
            timegate = PLSTMTimeGate()

            
        def add_timegate_params(gate, gate_name):
            
            """ Convenience function for adding layer parameters from a Gate
            instance. """

            # self.add_param
            return (self.add_param(gate.Period, (num_units, ), #  这里相当于给的 shape ，用initializer 类 进行初始化
                                                               #  这个地方为什么要留一个维度呢， 这三个参数都只是针对 h 和 c 进行的
                                   name="Period_{}".format(gate_name),
                                   trainable=learn_time_params[0]),
                    self.add_param(gate.Shift, (num_units, ),  
                                   name="Shift_{}".format(gate_name),
                                   trainable=learn_time_params[1]),
                    self.add_param(gate.On_End, (num_units, ),
                                   name="On_End_{}".format(gate_name),
                                   trainable=learn_time_params[2]))

        
        print('Learnableness: {}'.format(learn_time_params))

        #  这里 实际上  是创建  self--PLSTMLayer  类 的 变量  进行初始化
        #  初始化时间门的参数
        
        (self.period_timegate,
         self.shift_timegate,
         self.on_end_timegate) =  add_timegate_params(timegate, 'timegate')

        # Add in parameters from the supplied Gate instances
        #  初始化 输入门，  遗忘门，   输出门的参数
        (self.W_in_to_ingate,
         self.W_hid_to_ingate,
         self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate,
         self.W_hid_to_forgetgate,
         self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell,
         self.W_hid_to_cell,
         self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate,
         self.W_hid_to_outgate,
         self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')



        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        #  如果采用了窥视孔的连接，那么还要初始化窥视孔连接的参数

        #  窥视孔 在三个门 都有一个 W.cell 权重矩阵
        
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")



        # Setup initial values for the cell and the hidden units

        #  如果 cell_init 是一个 Layer 类，那么就把 Layer 类赋值给 这个 plstmlayer 的 cell_init
        
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
            
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            
            self.hid_init = hid_init
            
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hi d_init",
                trainable=learn_init, regularizable=False)

        if bn:  #  如果要做 bn-lstm 的话
            self.bn = lasagne.layers.BatchNormLayer(input_shape, axes=(0,1))  # create BN layer for correct input shape
            self.params.update(self.bn.params)  # make BN params your params
        else:
            self.bn = False


    def get_output_shape_for(self, input_shapes):
        
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        # 无论是否正在使用 mask input ，该层输入的形状将是input_shapes的第一个元素
        
        input_shape = input_shapes[0]

        
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened

        #  如果只输出最后的输出， 那么第二个维度就会被展平
        
        if self.only_return_final:
            
            return input_shape[0], self.num_units
        
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        
        else:
            
            # 否则就返回  不同批次， 不同 步数， 神经单元
            
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self,
                       inputs,
                       deterministic=False,
                       **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        
        input = inputs[0]

        
        # Retrieve the mask when it is supplied
        
        mask = None
        hid_init = None
        cell_init = None
        
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
            
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
            
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # PHASED LSTM: Define new input
        time_mat = inputs[self.time_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        if self.bn:
            input = self.bn.get_output_for(input)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        
        input = input.dimshuffle(1, 0, 2)

        
        # PHASED LSTM: Get shapes for time input and rearrange for the scan fn

        
        time_input = time_mat.dimshuffle(1,0)
        time_seq_len, time_num_batch = time_input.shape
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        
        W_in_stacked = T.concatenate(
                                    [self.W_in_to_ingate,
                                     self.W_in_to_forgetgate,
                                    self.W_in_to_cell,
                                     self.W_in_to_outgate],   axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
                                        [self.W_hid_to_ingate,
                                         self.W_hid_to_forgetgate,
                                         self.W_hid_to_cell,
                                         self.W_hid_to_outgate],    axis=1)

        # Stack biases into a (4*num_units) vector
        
        b_stacked = T.concatenate(
                                    [self.b_ingate,
                                     self.b_forgetgate,
                                     self.b_cell,
                                     self.b_outgate],    axis=0)

        # PHASED LSTM: If test time, off-phase means really shut.
        if deterministic:
            print('Using true off for testing.')
            off_slope = 0.0
        else:
            print('Using {} for off_slope.'.format(self.off_alpha))
            off_slope = self.off_alpha

        # PHASED LSTM: Pregenerate broadcast vars.
        #   Same neuron in different batches has same shift and period.  Also,
        #   precalculate the middle (on_mid) and end (on_end) of the open-phase
        #   ramp.
        ##  不同批次的相同神经元具有相同的移位和周期。 此外，预先计算开相斜坡的中间（on_mid）和结束（on_end）。

        
        shift_broadcast = self.shift_timegate.dimshuffle(['x',0])
        
        period_broadcast = T.abs_(self.period_timegate.dimshuffle(['x',0]))
        
        on_mid_broadcast = T.abs_(self.on_end_timegate.dimshuffle(['x',0])) * 0.5 * period_broadcast
        
        on_end_broadcast = T.abs_(self.on_end_timegate.dimshuffle(['x',0])) * period_broadcast

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input

        #  计算前向传播过程
        
        def step(input_n, time_input_n, cell_previous, hid_previous, *args):
            # input_n 是 
            
            if not self.precompute_input:
                
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Mix in new stuff
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        # PHASED LSTM: The actual calculation of the time gate
        #  计算时间门过程
        def calc_time_gate(time_input_n):
            # Broadcast the time across all units
            t_broadcast = time_input_n.dimshuffle([0,'x'])
            # Get the time within the period
            in_cycle_time = T.mod(t_broadcast + shift_broadcast, period_broadcast) # 这里不应该是 - 号吗
            # Find the phase
            is_up_phase = T.le(in_cycle_time, on_mid_broadcast)
            is_down_phase = T.gt(in_cycle_time, on_mid_broadcast)*T.le(in_cycle_time, on_end_broadcast)
            # Set the mask
            sleep_wake_mask = T.switch(is_up_phase, in_cycle_time/on_mid_broadcast,
                                T.switch(is_down_phase,
                                    (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                        off_slope*(in_cycle_time/period_broadcast)))

            return sleep_wake_mask

        # PHASED LSTM: Mask the updates based on the time phase
        #  基于时间门输出的信息去 掩盖某些神经元的更新
        
        def step_masked(input_n,
                        
                        time_input_n,
                        
                        mask_n,
                        
                        cell_previous,  #  前一时刻的细胞状态  (num_size)
                        hid_previous,  #   前一时刻的细胞状态   (num_size)
                        
                        *args):
            
            cell, hid = step(input_n,
                             
                             time_input_n,
                             
                             cell_previous,
                             hid_previous,
                             
                             *args)

            # Get time gate openness
            
            sleep_wake_mask = calc_time_gate(time_input_n)

            # Sleep if off, otherwise stay a bit on
            
            cell = sleep_wake_mask * cell + (1.-sleep_wake_mask) * cell_previous
            
            hid = sleep_wake_mask * hid + (1.-sleep_wake_mask) * hid_previous

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]
        #  如果 mask==0  那么就赋值上一时刻 c  ， h 的状态

        if mask is not None:
            #  忽然想到 如果 数据里面有缺失的话， 就可以把mask_n相应的位置弄为0,也就是说对于某个变量，这个time_step就不更新了
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')  #  'x' 的那一个维度为值就为1
        else:
            mask = T.ones_like(time_input).dimshuffle(0,1,'x')

        sequences = [input, time_input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))

        
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)


        non_seqs = [W_hid_stacked, self.period_timegate, self.shift_timegate, self.on_end_timegate]


        
        # The "peephole" weight matrices are only used when self.peepholes=True

        
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
