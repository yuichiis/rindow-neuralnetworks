<?php
namespace Rindow\NeuralNetworks\Layer;

# import re
# import string
# 
# import ml_dtypes
# import numpy as np
# 
# from keras.src import activations
# from keras.src import constraints
# from keras.src import dtype_policies
# from keras.src import initializers
# from keras.src import ops
# from keras.src import quantizers
# from keras.src import regularizers
# from keras.src.api_export import keras_export
# from keras.src.layers.input_spec import InputSpec
# from keras.src.layers.layer import Layer


use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
/*
        """A layer that uses `einsum` as the backing computation.
    
        This layer can perform einsum calculations of arbitrary dimensionality.
    
        Args:
            equation: An equation describing the einsum to perform.
                This equation must be a valid einsum string of the form
                `ab,bc->ac`, `...ab,bc->...ac`, or
                `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
                axis expression sequence.
            output_shape: The expected shape of the output tensor
                (excluding the batch dimension and any dimensions
                represented by ellipses). You can specify `None` for any dimension
                that is unknown or can be inferred from the input shape.
            activation: Activation function to use. If you don't specify anything,
                no activation is applied
                (that is, a "linear" activation: `a(x) = x`).
            bias_axes: A string containing the output dimension(s)
                to apply a bias to. Each character in the `bias_axes` string
                should correspond to a character in the output portion
                of the `equation` string.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to the `kernel` weights
                matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            kernel_constraint: Constraint function applied to the `kernel` weights
                matrix.
            bias_constraint: Constraint function applied to the bias vector.
            lora_rank: Optional integer. If set, the layer's forward pass
                will implement LoRA (Low-Rank Adaptation)
                with the provided rank. LoRA sets the layer's kernel
                to non-trainable and replaces it with a delta over the
                original kernel, obtained via multiplying two lower-rank
                trainable matrices
                (the factorization happens on the last dimension).
                This can be useful to reduce the
                computation cost of fine-tuning large dense layers.
                You can also enable LoRA on an existing
                `EinsumDense` layer by calling `layer.enable_lora(rank)`.
            **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    
        Examples:
    
        **Biased dense layer with einsums**
    
        This example shows how to instantiate a standard Keras dense layer using
        einsum operations. This example is equivalent to
        `keras.layers.Dense(64, use_bias=True)`.
    
        >>> layer = keras.layers.EinsumDense("ab,bc->ac",
        ...                                       output_shape=64,
        ...                                       bias_axes="c")
        >>> input_tensor = keras.Input(shape=[32])
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        (None, 64)
    
        **Applying a dense layer to a sequence**
    
        This example shows how to instantiate a layer that applies the same dense
        operation to every element in a sequence. Here, the `output_shape` has two
        values (since there are two non-batch dimensions in the output); the first
        dimension in the `output_shape` is `None`, because the sequence dimension
        `b` has an unknown shape.
    
        >>> layer = keras.layers.EinsumDense("abc,cd->abd",
        ...                                       output_shape=(None, 64),
        ...                                       bias_axes="d")
        >>> input_tensor = keras.Input(shape=[32, 128])
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        (None, 32, 64)
    
        **Applying a dense layer to a sequence using ellipses**
    
        This example shows how to instantiate a layer that applies the same dense
        operation to every element in a sequence, but uses the ellipsis notation
        instead of specifying the batch and sequence dimensions.
    
        Because we are using ellipsis notation and have specified only one axis, the
        `output_shape` arg is a single value. When instantiated in this way, the
        layer can handle any number of sequence dimensions - including the case
        where no sequence dimension exists.
    
        >>> layer = keras.layers.EinsumDense("...x,xy->...y",
        ...                                       output_shape=64,
        ...                                       bias_axes="y")
        >>> input_tensor = keras.Input(shape=[32, 128])
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        (None, 32, 64)
        """
*/
class EinsumDense extends AbstractLayer
{
    use GenericUtils;
    protected int $units;
    protected bool $useBias;
    protected mixed $kernelInitializer;
    protected mixed $biasInitializer;
    protected ?string $kernelInitializerName;
    protected ?string $biasInitializerName;

    protected ?NDArray $kernel=null;
    protected NDArray $bias;
    protected NDArray $dKernel;
    protected NDArray $dBias;

    protected string $equation;
    protected array $partial_output_shape;
    protected array $full_output_shape;
    protected ?string $bias_axes;
    protected ?int $lora_rank;
    protected bool $lora_enabled;
    protected string $dInputsBackwardEquation;
    protected string $dKernelBackwardEquation;
    
    //protected $inputs;

    //"""Quantization-related (int8 and float8) methods"""
    private string $QUANTIZATION_MODE_ERROR_TEMPLATE = 
        "Invalid quantization mode. Expected one of ".
        "{dtype_policies.QUANTIZATION_MODES}. ".
        "Received: quantization_mode={mode}";

    public function __construct(
        object $backend,
        string $equation,
        int|array $output_shape,
        array $input_shape=null,
        string|object $activation=null,
        string $bias_axes=null,
        string|object $kernel_initializer=null,
        string|object $bias_initializer=null,
        //string|object $kernel_regularizer=null,
        //string|object $bias_regularizer=null,
        //string|object $kernel_constraint=null,
        //string|object $bias_constraint=null,
        int $lora_rank=null,
        string $name=null,
        //**kwargs,
    )
    {
        $kernel_initializer ??= 'glorot_uniform';
        $bias_initializer ??= 'zeros';

        $this->equation = $equation;
        if(is_int($output_shape)) {
            $output_shape = [$output_shape];
        }
        parent::__construct($backend);
        $K = $backend;
        $this->inputShape = $input_shape;
        $this->partial_output_shape = $output_shape;
        $this->bias_axes = $bias_axes;
        $this->useBias = ($this->bias_axes!==null) ? true : false;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $this->toStringName($kernel_initializer);
        $this->biasInitializerName = $this->toStringName($bias_initializer);
        $this->lora_rank = $lora_rank;
        $this->lora_enabled = false;
        $this->initName($name,'einsumdense');
        $this->allocateWeights($this->useBias?2:1);
        $this->setActivation($activation);
    }
    
    public function build(mixed $variable=null, array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($variable);
        //echo "einsum inputShape arg=(".implode(',',$inputShape).")\n";
        //echo "einsum partial_output_shape arg=(".implode(',',$this->partial_output_shape).")\n";

        [
            $kernel_shape,
            $bias_shape,
            $full_output_shape,
            $backward_dinput_equation,
            $backward_dkernel_equation
        ] = $this->analyze_einsum_string(
            $this->equation,
            $this->bias_axes,
            $inputShape,
            $this->partial_output_shape,
        );
        $this->dInputsBackwardEquation  = $backward_dinput_equation;
        $this->dKernelBackwardEquation  = $backward_dkernel_equation;
        $this->full_output_shape = $full_output_shape;
        # `self._int8_build` needs `self.input_spec`
        # $this->input_spec = new InputSpec(ndim:count($input_shape));
        # We use `self._dtype_policy` to check to avoid issues in torch dynamo
        # $is_quantized = ($this->dtype_policy instanceof QuantizedDTypePolicy);
        #if($is_quantized) {
        #    $this->quantized_build(
        #        $input_shape, mode:$this->dtype_policy->quantization_mode
        #    );
        #}
        #if(!$is_quantized || $this->dtype_policy->quantization_mode != NDArray::int8) {
            # If the layer is quantized to int8, `self._kernel` will be added
            # in `self._int8_build`. Therefore, we skip it here.
            #$this->kernel = $this->add_weight(
            #    name:"kernel",
            #    shape:$kernel_shape,
            #    initializer:$this->kernel_initializer,
            #    //regularizer:$this->kernel_regularizer,
            #    //constraint:$this->kernel_constraint,
            #    dtype:$this->dtype,
            #    trainable:true,
            #);
        #}
        #if($bias_shape) {
        #    $this->bias = $this->add_weight(
        #        name:"bias",
        #        shape:$bias_shape,
        #        initializer:$this->bias_initializer,
        #        //regularizer:$this->bias_regularizer,
        #        //constraint:$this->bias_constraint,
        #        dtype:$this->dtype,
        #        trainable:true,
        #    );
        #} else {
        #    $this->bias = null;
        #}
        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
                if($this->useBias) {
                    $this->bias = $sampleWeights[1];
                }
            } else {
                $this->kernel = $kernelInitializer(
                    $kernel_shape,
                    $kernel_shape);
                if($this->useBias) {
                    $this->bias = $biasInitializer(
                        $bias_shape);
                }
            }
        }
        $this->built = true;
        if($this->lora_rank) {
            $this->enable_lora($this->lora_rank);
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        $output_shape = $full_output_shape;
        array_shift($output_shape);
        $this->outputShape = $output_shape;
        $this->syncWeightVariables();
    }

    public function getParams() : array
    {
        if($this->useBias) {
            return [$this->kernel,$this->bias];
        } else {
            return [$this->kernel];
        }
    }

    public function getGrads() : array
    {
        if($this->useBias) {
            return [$this->dKernel,$this->dBias];
        } else {
            return [$this->dKernel];
        }
    }

    public function kernel() : NDArray
    {
        if(!$this->built) {
            throw new LogicException(
                "You must build the layer before accessing `kernel`."
            );
        }
        if($this->lora_enabled) {
            return $K->add($this->kernel, $K->matmul(
                $this->lora_kernel_a, $this->lora_kernel_b
            ));
        }
        return $this->kernel;
    }
    
    public function compute_output_shape() : array
    {
        return $this->full_output_shape;
    }
    
    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $outputs = $K->einsum($this->equation, $inputs, $this->kernel());
        if($this->useBias) {
            //echo "useBias in forward in EinsumDense\n";
            //echo "outputs=(".implode(',',$outputs->shape()).")\n";
            //echo "bias=(".implode(',',$this->bias->shape()).")\n";
            $K->update_add($outputs,$this->bias);
        }
        if($this->activation) {
            $container->activation = new \stdClass();
            $outputs = $this->activation->forward($container->activation,$outputs,training:$training);
        }
        return $outputs;
    }
    
    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        if($this->activation) {
            $dOutputs = $this->activation->backward($container->activation,$dOutputs);
        }
        $dInputs = $K->einsum($this->dInputsBackwardEquation, $dOutputs, $this->kernel());

        // update params
        $dKernel = $K->einsum($this->dKernelBackwardEquation, $dOutputs, $container->inputs);
        $K->copy($dKernel,$this->dKernel);
        if($this->useBias) {
            $biasFlatSize = (int)array_product($this->dBias->shape());
            $dOutputsFlatSize = (int)array_product($dOutputs->shape());
            $dOutputsFlat = $dOutputs->reshape([intdiv($dOutputsFlatSize,$biasFlatSize),$biasFlatSize]);
            $dBiasFlat = $this->dBias->reshape([$biasFlatSize]);
            $K->sum($dOutputsFlat, axis:0, output:$dBiasFlat);
        }

        return $dInputs;
    }

    private function enable_lora(
        int $rank,
        string|object $a_initializer=null,
        string|object $b_initializer=null
    ) : void
    {
        $a_initializer ??= "he_uniform";
        $b_initializer ??= "zeros";
        if($this->kernel_constraint) {
            throw new InvalidArgumentException(
                "Lora is incompatible with kernel constraints. ".
                "In order to enable lora on this layer, remove the ".
                "`kernel_constraint` argument."
            );
        }
        if(!$this->built) {
            throw new InvalidArgumentException(
                "Cannot enable lora on a layer that isn't yet built."
            );
        }
        if($this->lora_enabled) {
            throw new InvalidArgumentException(
                "lora is already enabled. ".
                "This can only be done once per layer."
            );
        }
        $this->tracker->unlock();
        $this->lora_kernel_a = $this->add_weight(
            name:"lora_kernel_a",
            shape:[array_merge($this->kernel()->shape()[R(null,-1)],[$rank])],
            initializer:$K->getInitializers($a_initializer),
            //regularizer:$this->kernel_regularizer,
        );
        $this->lora_kernel_b = $this->add_weight(
            name:"lora_kernel_b",
            shape:[$rank, $this->kernel()->shape()[-1]],
            initializer:$K->getInitializers($b_initializer),
            //regularizer:$this->kernel_regularizer,
        );
        $this->kernel->trainable = false;
        $this->tracker->lock();
        $this->lora_enabled = true;
        $this->lora_rank = $rank;
    }
    
    public function save_own_variables(array $store) : void
    {
        # Do nothing if the layer isn't yet built
        if(!$this->built) {
            return;
        }
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        [$kernel_value, $kernel_scale] = $this->get_kernel_with_merged_lora();
        $target_variables = [$kernel_value];
        if($this->bias) {
            $target_variables->append($this->bias);
        }
        #if($this->dtype_policy instanceof QuantizedDTypePolicy) {
        #    $mode = $this->dtype_policy->quantization_mode;
        #    if($mode == NDArray::int8) {
        #        $target_variables->append($kernel_scale);
        #    } elseif($mode == NDArray::float8) {
        #        $target_variables->append($this->inputs_scale);
        #        $target_variables->append($this->inputs_amax_history);
        #        $target_variables->append($this->kernel_scale);
        #        $target_variables->append($this->kernel_amax_history);
        #        $target_variables->append($this->outputs_grad_scale);
        #        $target_variables->append($this->outputs_grad_amax_history);
        #    } else {
        #        throw new NotImplementedError(
        #            $this->QUANTIZATION_MODE_ERROR_TEMPLATE->format($mode)
        #        );
        #    }
        #}
        foreach($target_variables as $i => $variable) {
            $store[strval($i)] = $variable;
        }
    }
    
    private function load_own_variables($store) : void
    {
        if(!$this->lora_enabled) {
            $this->check_load_own_variables($store);
        }
        # Do nothing if the layer isn't yet built
        if(!$this->built) {
            return;
        }
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        $target_variables = [$this->kernel];
        if($this->bias) {
            $target_variables->append($this->bias);
        }
        #if($this->dtype_policy instanceof QuantizedDTypePolicy) {
        #    $mode = $this->dtype_policy->quantization_mode;
        #    if($mode == NDArray::int8) {
        #        $target_variables->append($this->kernel_scale);
        #    } elseif($mode == NDArray::float8) {
        #        $target_variables->append($this->inputs_scale);
        #        $target_variables->append($this->inputs_amax_history);
        #        $target_variables->append($this->kernel_scale);
        #        $target_variables->append($this->kernel_amax_history);
        #        $target_variables->append($this->outputs_grad_scale);
        #        $target_variables->append($this->outputs_grad_amax_history);
        #    } else {
        #        throw new NotImplementedError(
        #            $this->QUANTIZATION_MODE_ERROR_TEMPLATE->format($mode)
        #        );
        #    }
        #}
        foreach($target_variables as $i => $variable) {
            $variable->assign($store[strval($i)]);
        }
        if($this->lora_enabled) {
            $this->lora_kernel_a->assign($K->zeros($this->lora_kernel_a->shape()));
            $this->lora_kernel_b->assign($K->zeros($this->lora_kernel_b->shape()));
        }
    }

    public function get_config() : array
    {
        $config = [
            "output_shape"=>$this->partial_output_shape,
            "equation"=>$this->equation,
            "activation"=>$this->activationName,
            "bias_axes"=>$this->bias_axes,
            "kernel_initializer"=>$this->kernel_initializer_name,
            "bias_initializer"=>$this->bias_initializer_name,
            //"kernel_regularizer"=>$this->kernel_regularizer_name,
            //"bias_regularizer"=>$this->bias_regularizer_name,
            //"activity_regularizer"=>$this->activity_regularizer_name,
            //"kernel_constraint"=>$this->kernel_constraint_name,
            //"bias_constraint"=>$this->bias_constraint_name,
        ];
        if($this->lora_rank){
            $config["lora_rank"] = $this->lora_rank;
        }
        return $config;
    }
    
    private function check_load_own_variables(array $store) : void
    {
        $all_vars = array_merge($this->trainable_variables, $this->non_trainable_variables);
        if(count(array_keys($store)) != count($all_vars)) {
            if(count($all_vars) == 0 && !$this->built) {
                throw new InvalidArgumentException(
                    "Layer '{self.name}' was never built ".
                    "and thus it doesn't have any variables. ".
                    "However the weights file lists {len(store.keys())} ".
                    "variables for this layer.\n".
                    "In most cases, this error indicates that either:\n\n".
                    "1. The layer is owned by a parent layer that ".
                    "implements a `build()` method, but calling the ".
                    "parent's `build()` method did NOT create the state of ".
                    "the child layer '{self.name}'. A `build()` method ".
                    "must create ALL state for the layer, including ".
                    "the state of any children layers.\n\n".
                    "2. You need to implement ".
                    "the `def build_from_config(self, config)` method ".
                    "on layer '{self.name}', to specify how to rebuild ".
                    "it during loading. ".
                    "In this case, you might also want to implement the ".
                    "method that generates the build config at saving time, ".
                    "`def get_build_config(self)`. ".
                    "The method `build_from_config()` is meant ".
                    "to create the state ".
                    "of the layer (i.e. its variables) upon deserialization."
                );
            }
            throw new InvalidArgumentException(
                "Layer '{self.name}' expected {len(all_vars)} variables, ".
                "but received ".
                "{len(store.keys())} variables during loading. ".
                "Expected: {[v.name for v in all_vars]}"
            );
        }
    }
    
    private function get_kernel_with_merged_lora() : array
    {
        #if($this->dtype_policy instanceof QuantizedDTypePolicy) {
        #    $kernel_value = $this->kernel;
        #    $kernel_scale = $this->kernel_scale;
        #    if($this->lora_enabled) {
        #        # Dequantize & quantize to merge lora weights into int8 kernel
        #        # Note that this is a lossy compression
        #        $kernel_value = $K->divide($kernel_value, $kernel_scale);
        #        $kernel_value = $K->add(
        #            $kernel_value,
        #            $K->matmul($lora_kernel_a, $lora_kernel_b),
        #        );
        #        [$kernel_value, $kernel_scale] = $K->abs_max_quantize(
        #            $kernel_value, axis:$this->kernel_reduced_axes
        #        );
        #        $kernel_scale = $K->transpose(
        #            $kernel_scale, $this->kernel_transpose_axes
        #        );
        #        if($this->kernel_expand_axes) {
        #            $kernel_scale = $K->expand_dims(
        #                $kernel_scale, axis:$this->kernel_expand_axes
        #            );
        #        }
        #        if($this->kernel_squeeze_axes) {
        #            $kernel_scale = $K->squeeze(
        #                $kernel_scale, axis:$this->kernel_squeeze_axes
        #            );
        #        }
        #    }
        #} else {
            $kernel_value = $this->kernel;
            $kernel_scale = null;
        #}
        return [$kernel_value, $kernel_scale];
    }

    /*
        """Analyzes an einsum string to determine the required weight shape."""

        return [
            $kernel_shape,
            $bias_shape,
            $full_output_shape,
            $backward_dinput_equation,
            $backward_dkernel_equation
        ]
    */
    private function analyze_einsum_string(
        string $equation,
        ?string $bias_axes,
        array $input_shape,
        array $output_shape
    ) : array
    {
        //echo "equation=$equation\n";
        //echo "einsum analyze_einsum_string output_shape arg=(".implode(',',$output_shape).")\n";
        //$dot_replaced_string = $re->sub("\.\.\.", "0", $equation);
        $dot_replaced_string = str_replace("...", "0", $equation);
        # This is the case where no ellipses are present in the string.
        //$split_string = $re->match(
        //    "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", $dot_replaced_string
        //);
        preg_match(
            "/([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)/",
            $dot_replaced_string,
            $split_string
        );
        if($bias_axes!==null) {
            $bias_axes = str_split($bias_axes);
        }
        if($split_string) {
            [$dmy,$input_chrs,$weight_chrs,$output_chrs] = $split_string;
            if(count($input_shape)+1!=strlen($input_chrs)) {
                throw new InvalidArgumentException('Unmatch rank of input_shape and input spec in equation');
            }
            if(count($output_shape)+1!=strlen($output_chrs)) {
                throw new InvalidArgumentException('Unmatch rank of output_shape and output spec in equation');
            }
            $results = $this->analyze_split_string(
                $split_string, $bias_axes, $input_shape, $output_shape
            );
            $backward_dinput_script  = "{$output_chrs},{$weight_chrs}->{$input_chrs}";
            $backward_dkernel_script = "{$output_chrs},{$input_chrs}->{$weight_chrs}";
            return array_merge($results,[$backward_dinput_script,$backward_dkernel_script]);
        }
    
        # This is the case where ellipses are present on the left.
        //$split_string = $re->match(
        //    "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", $dot_replaced_string
        //);
        preg_match(
            "/0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)/",
            $dot_replaced_string,
            $split_string
        );
        if($split_string) {
            [$dmy,$input_chrs,$weight_chrs,$output_chrs] = $split_string;
            $results = $this->analyze_split_string(
                $split_string, $bias_axes, $input_shape, $output_shape, $left_elided=true
            );
            $backward_dinput_script  = "...{$output_chrs},{$weight_chrs}->...{$input_chrs}";
            $backward_dkernel_script = "...{$output_chrs},...{$input_chrs}->{$weight_chrs}";
            return array_merge($results,[$backward_dinput_script,$backward_dkernel_script]);
        }
    
        # This is the case where ellipses are present on the right.
        //$split_string = $re->match(
        //    "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", $dot_replaced_string
        //);
        preg_match(
            "/([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0/",
            $dot_replaced_string,
            $split_string
        );
        if($split_string) {
            [$dmy,$input_chrs,$weight_chrs,$output_chrs] = $split_string;
            $results = $this->analyze_split_string(
                $split_string, $bias_axes, $input_shape, $output_shape
            );
            $backward_dinput_script  = "{$output_chrs}...,{$weight_chrs}->{$input_chrs}...";
            $backward_dkernel_script = "{$output_chrs}...,{$input_chrs}...->{$weight_chrs}";
            return array_merge($results,[$backward_dinput_script,$backward_dkernel_script]);
        }
    
        throw new InvalidArgumentException(
            "Invalid einsum equation '{$equation}'. Equations must be in the form ".
            "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
        );
    }
    

    /*
            """Analyze an pre-split einsum string to find the weight shape."""
     */
    private function analyze_split_string(
        array $split_string,
        ?array $bias_axes,
        ?array $input_shape,
        array $output_shape,
        bool $left_elided=null
    ) : array
    {
        //echo "einsum analyze_split_string input_shape arg=(".implode(',',$input_shape).")\n";
        //echo "einsum analyze_split_string output_shape arg=(".implode(',',$output_shape).")\n";
        $left_elided ??= false;
        $input_spec = str_split($split_string[1]);
        $weight_spec = str_split($split_string[2]);
        $output_spec = str_split($split_string[3]);

        array_unshift($input_shape, 1);  // add batch shape
        //echo "input_shape rank=".count($input_shape)."\n";
        //echo "input_spec rank=".count($input_spec)."\n";
        $elided = count($input_shape) - count($input_spec);

        array_unshift($output_shape, $input_shape[0]);
        //echo "einsum analyze_split_string output_shape array_unshift=(".implode(',',$output_shape).")\n";
        //echo "elided=";var_dump($elided);
        //echo "left_elided=";var_dump($left_elided);
        if($elided > 0 && $left_elided) {
            $top = array_shift($output_shape);
            for($i=1; $i<$elided; $i++) {
                # We already inserted the 0th input dimension at dim 0, so we need
                # to start at location 1 here.
                array_unshift($output_shape,$input_shape[$i]);
            }
            array_unshift($output_shape,$top);
        } elseif($elided > 0 && !$left_elided) {
            $count = count($input_shape);
            for($i=count($input_shape) - $elided; $i<$count; $i++) {
                array_push($output_shape,$input_shape[$i]);
            }
        }
        //echo "einsum analyze_split_string output_shape format=(".implode(',',$output_shape).")\n";

        if($left_elided) {
            # If we have beginning dimensions elided, we need to use negative
            # indexing to determine where in the input dimension our values are.
            $input_dim_map = [];
            foreach($input_spec as $i=>$dim) {
                $pos = ($i + $elided) - count($input_shape);
                $pos = ($pos<0) ? count($input_shape)+$pos : $pos;
                $input_dim_map[$dim] = $pos;
            }
            # Because we've constructed the full output shape already, we don't need
            # to do negative indexing.
            $output_dim_map = [];
            foreach($output_spec as $i=>$dim) {
                $output_dim_map[$dim] = $i + $elided;
            }
        } else {
            $input_dim_map = array_flip($input_spec);
            $output_dim_map = array_flip($output_spec);
        }
    
        foreach($input_spec as $dim) {
            $input_shape_at_dim = $input_shape[$input_dim_map[$dim]];
            if(in_array($dim,$output_dim_map)) {
                $output_shape_at_dim = $output_shape[$output_dim_map[$dim]];
                if(
                    $output_shape_at_dim !==null &&             // NOT free dim
                    $output_shape_at_dim != $input_shape_at_dim // fixed dim
                ) {
                    throw new InvalidArgumentException(
                        "Input shape and output shape do not match at shared ".
                        "dimension '{$dim}'. Input shape is {$input_shape_at_dim}, ".
                        "and output shape ".
                        "is ".$output_shape[$output_dim_map[$dim]]."."
                    );
                }
            }
        }
    
        foreach($output_spec as $dim) {
            if(!in_array($dim,$input_spec) && !in_array($dim,$weight_spec)) {
                throw new InvalidArgumentException(
                    "Dimension '{$dim}' was specified in the output ".
                    "'".implode(',',$output_spec)."' but has no corresponding dim in the input ".
                    "spec '".implode(',',$input_spec)."' or weight spec '".implode(',',$output_spec)."'"
                );
            }
        }
    
        $weight_shape = [];
        foreach($weight_spec as $dim) {
            if(array_key_exists($dim,$input_dim_map)) {
                array_push($weight_shape,$input_shape[$input_dim_map[$dim]]);
            } elseif(array_key_exists($dim,$output_dim_map)) {
                array_push($weight_shape,$output_shape[$output_dim_map[$dim]]);
            } else {
                throw new InvalidArgumentException(
                    "Weight dimension '{$dim}' did not have a match in either ".
                    "the input spec '".implode(',',$input_spec)."' or the output ".
                    "spec '".implode(',',$output_spec)."'. For this layer, the weight must ".
                    "be fully specified."
                );
            }
        }

        if($bias_axes) {
            $num_left_elided = ($left_elided) ? $elided : 0;
            $idx_map = [];
            foreach($output_spec as $i=>$char) {
                $idx_map[$char] = $output_shape[$i + $num_left_elided];
            }
    
            foreach($bias_axes as $char) {
                if(!in_array($char,$output_spec)) {
                    throw new InvalidArgumentException(
                        "Bias dimension '{$char}' was requested, but is not part ".
                        "of the output spec '".implode(',',$output_spec)."'"
                    );
                }
            }
            $flip_output_spec = array_flip($output_spec);
            $first_bias_location = min(
                array_map(fn($char)=>$flip_output_spec[$char],$bias_axes)
            );
            $bias_output_spec = array_slice($output_spec,$first_bias_location);
    
            $bias_shape = array_map(
                    fn($char)=>(in_array($char,$bias_axes))?$idx_map[$char]:1, 
                    $bias_output_spec
            );
    
            if(!$left_elided) {
                for($i=0;$i<$elided;++$i) {
                    $bias_shape[] = 1;
                }
            }
        } else {
            $bias_shape = null;
        }
    
        return [$weight_shape, $bias_shape, $output_shape];
    }
    
/*
    private function get_specs($equation, $input_shape) : array
    {
        $possible_labels = string::ascii_letters;
        $dot_replaced_string = $re->sub("\.\.\.", "0", $equation);

        # This is the case where no ellipses are present in the string.
        $split_string = $re->match(
            "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", $dot_replaced_string
        );
        if($split_string) {
            $input_spec = $split_string->group(1);
            $weight_spec = $split_string->group(2);
            $output_spec = $split_string->group(3);
            return [$input_spec, $weight_spec, $output_spec];
        }

        # This is the case where ellipses are present on the left.
        $split_string = $re->match(
            "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", $dot_replaced_string
        );
        if($split_string) {
            $input_spec = $split_string->group(1);
            $weight_spec = $split_string->group(2);
            $output_spec = $split_string->group(3);
            $elided = count($input_shape) - count($input_spec);
            $possible_labels = sorted(
                set($possible_labels)
                - set($input_spec)
                - set($weight_spec)
                - set($output_spec)
            );
            # Pad labels on the left to `input_spec` and `output_spec`
            for($i=0;$i<$elided;++$i) {
                $input_spec = $possible_labels[$i] + $input_spec;
                $output_spec = $possible_labels[$i] + $output_spec;
            }
            return [$input_spec, $weight_spec, $output_spec];
        }
        # This is the case where ellipses are present on the right.
        $split_string = $re->match(
            "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", $dot_replaced_string
        );
        if($split_string) {
            $input_spec = $split_string->group(1);
            $weight_spec = $split_string->group(2);
            $output_spec = $split_string->group(3);
            $elided = count($input_shape) - count($input_spec);
            $possible_labels = sorted(
                set($possible_labels)
                - set($input_spec)
                - set($weight_spec)
                - set($output_spec)
            );
            # Pad labels on the right to `input_spec` and `output_spec`
            for($i=0;$i<$elided;++$i) {
                $input_spec = $input_spec + $possible_labels[$i];
                $output_spec = $output_spec + $possible_labels[$i];
            }
            return [$input_spec, $weight_spec, $output_spec];
        }

        throw new InvalidArgumentException(
            "Invalid einsum equation '{equation}'. Equations must be in the ".
            "form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
        );
    }

    private function analyze_quantization_info($equation, $input_shape)
    {
        [$input_spec, $weight_spec, $output_spec] = $this->get_specs($equation, $input_shape);
    
        # Determine the axes that should be reduced by the quantizer
        $input_reduced_axes = [];
        $weight_reduced_axes = [];
        foreach($input_spec as $i=>$label) {
            $index = $output_spec[$label];
            if($index == -1) {
                $input_reduced_axes[] = $i;
            }
        }
        foreach($weight_spec as $i=>$label) {
            $index = $output_spec[$label];
            if($index == -1) {
                $weight_reduced_axes[] = $i;
            }
        }
    
        # Determine the axes of `ops.expand_dims`
        $input_expand_axes = [];
        $weight_expand_axes = [];
        foreach($output_spec as $i=>$label) {
            $index_input = $input_spec[$label];
            $index_weight = $weight_spec[$label];
            if($index_input == -1) {
                $input_expand_axes[] = $i;
            }
            if($index_weight == -1) {
                $weight_expand_axes[] = $i;
            }
        }
    
        # Determine the axes of `ops.transpose`
        $input_transpose_axes = [];
        $weight_transpose_axes = [];
        foreach($output_spec as $i=>$label) {
            $index_input = $input_spec[$label];
            $index_weight = $weight_spec[$label];
            if($index_input != -1) {
                $input_transpose_axes[] = $index_input;
            }
            if($index_weight != -1) {
                $weight_transpose_axes[] = $index_weight;
            }
        }
        # Postprocess the information:
        # 1. Add dummy axes (1) to transpose_axes
        # 2. Add axis to squeeze_axes if 1. failed
        $input_squeeze_axes = [];
        $weight_squeeze_axes = [];
        foreach($input_reduced_axes as $ori_index) {
            try {
                $index = $input_expand_axes->pop(0);
            } catch(IndexError $e) {
                $input_squeeze_axes[] = $ori_index;
            }
            $input_transpose_axes->insert($index, $ori_index);
        }
        foreach($weight_reduced_axes as $ori_index) {
            try {
                $index = $weight_expand_axes->pop(0);
            } catch(IndexError $e) {
                $weight_squeeze_axes[] = $ori_index;
            }
            $weight_transpose_axes->insert($index, $ori_index);
        }
        # Prepare equation for `einsum_with_inputs_gradient`
        $custom_gradient_equation = "{output_spec},{weight_spec}->{input_spec}";
        $tmp = $weight_transpose_axes;
        asort($tmp);
        $weight_reverse_transpose_axes = array_keys($tmp);
        return [
            $input_reduced_axes,
            $weight_reduced_axes,
            $input_transpose_axes,
            $weight_transpose_axes,
            $input_expand_axes,
            $weight_expand_axes,
            $input_squeeze_axes,
            $weight_squeeze_axes,
            $custom_gradient_equation,
            $weight_reverse_transpose_axes,
        ];
    }
*/
}
