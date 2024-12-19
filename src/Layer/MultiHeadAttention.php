<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Layer\Activation;
use Rindow\NeuralNetworks\Layer\Dropout;

/*
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need"
    [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as `value_dim` can take
    a linear projection and return.

    Args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        value_dim: Size of each attention head for value.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        output_shape: The expected shape of an output tensor, besides the batch
            and sequence dims. If not specified, projects back to the query
            feature dim (the query input's last dimension).
        attention_axes: axes over which the attention is applied. `None` means
            attention over all axes, but batch, heads, and features.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.

    Call arguments:
        query: Query tensor of shape `(B, T, dim)`, where `B` is the batch size,
            `T` is the target sequence length, and dim is the feature dimension.
        value: Value tensor of shape `(B, S, dim)`, where `B` is the batch size,
            `S` is the source sequence length, and dim is the feature dimension.
        key: Optional key tensor of shape `(B, S, dim)`. If not given, will
            use `value` for both `key` and `value`, which is the most common
            case.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        returnAttentionScores: A boolean to indicate whether the output should
            be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Will go with either using the training mode of the parent
            layer/model, or `False` (inference) if there is no parent layer.
        useCausalMask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: The result of the computation, of shape `(B, T, E)`,
            where `T` is for target sequence shapes and `E` is the query input
            last dimension if `output_shape` is `None`. Otherwise, the
            multi-head outputs are projected to the shape specified by
            `output_shape`.
        attention_scores: (Optional) multi-head attention coefficients over
            attention axes.
    """
*/
class MultiHeadAttention extends AbstractAttentionLayer
{
    protected bool $supportsMasking;
    protected int $numHeads;
    protected int $keyDim;
    protected int $valueDim;
    protected float $dropout;
    protected bool $useBias;
    protected float $inverse_sqrt_key_dim;
    protected mixed $kernelInitializer;
    protected mixed $biasInitializer;
    protected ?string $kernelInitializerName;
    protected ?string $biasInitializerName;
    /** @var array<int> $attention_axes */
    protected ?array $attention_axes;
    protected Layer $query_dense;
    protected Layer $key_dense;
    protected Layer $value_dense;
    protected string $dot_product_equation;
    protected string $backward_dot_product_key_equation;
    protected string $backward_dot_product_query_equation;
    protected string $combine_equation;
    protected string $backward_combine_scores_equation;
    protected string $backward_combine_value_equation;
    protected Layer $softmax_layer;
    protected Layer $dropout_layer;
    /** @var array<int> $partial_output_shape */
    protected ?array $partial_output_shape;
    protected Layer $output_dense;



    protected bool $useScale;
    protected bool $doNotExpandMask;
    protected NDArray $scale;
    protected NDArray $dScale;
    /** @var array<int> $scoresShape */
    protected $scoresShape;
    /** @var array<bool> $unbackpropagatables */
    protected ?array $unbackpropagatables = null;

    //protected $returnAttentionScores;

    //protected $query;
    //protected $value;
    //protected $key;
    //protected $attentionWeight;

    /**
     * @param array<array<int>> $input_shapes
     * @param array<array<int>> $output_shape
     * @param array<int> $attention_axes
     */
    public function __construct(
        object $backend,
        int $num_heads,
        int $key_dim,
        int $value_dim=null,
        float $dropout=null,
        bool $use_bias=null,
        array $input_shapes=null,
        array $output_shape=null,
        int|array $attention_axes=null,
        string|object $kernel_initializer=null,
        string|object $bias_initializer=null,
        string $name=null,
    )
    {
        $value_dim ??= $key_dim;
        $dropout ??= 0.0;
        $use_bias ??= true;
        $kernel_initializer ??= 'glorot_uniform';
        $bias_initializer ??= 'zeros';

        parent::__construct($backend);
        $K = $backend;
        $this->supportsMasking = true;
        $this->numHeads = $num_heads;
        $this->keyDim = $key_dim;
        $this->inverse_sqrt_key_dim = 0.0;
        $this->valueDim = $value_dim;
        $this->dropout = $dropout;
        $this->useBias = $use_bias;
        $this->inputShape = $input_shapes;
        $this->partial_output_shape = $output_shape;  // without batch and sequence
        if($attention_axes!==null) {
            if(is_int($attention_axes)) {
                $attention_axes = [$attention_axes];
            }
        }
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $this->toStringName($kernel_initializer);
        $this->biasInitializerName = $this->toStringName($bias_initializer);
        $this->attention_axes = $attention_axes;
        $this->initName($name,'multiheadattention');
        $this->allocateWeights($this->useBias?2:1);
    }
    
    /*
        """Builds layers and variables.

        Args:
            query_shape: Shape of the `query` tensor.
            value_shape: Shape of the `value` tensor.
            key: Optional shape of the `key` tensor.
        """
    */
    public function build(mixed $variables=null, array $sampleWeights=null) : void
    {
        //echo "============== build =============\n";
        $K = $this->backend;
        $inputShapes = $this->normalizeInputShapes($variables);
        if(count($inputShapes)!=2&&count($inputShapes)!=3) {
            throw new InvalidArgumentException('num of inputs must be 2 or 3: inputs is '.count($inputShapes));
        }
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)||count($shape)<2) {
                $type = '['.implode(',',$shape).']';
                throw new InvalidArgumentException('input_shapes must be the list of shape: '.$type.' included in #'.$idx.'.');
            }
        }
        $query_shape = $inputShapes[0];  // Query
        $value_shape = $inputShapes[1]; // Value
        if(count($inputShapes)==3) {
            if($inputShapes[1]!=$inputShapes[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
            $key_shape = $inputShapes[2]; // Key;
        }

        $key_shape ??= $value_shape;

        //echo "numHeads=";var_dump($this->numHeads);
        //echo "keyDim=";var_dump($this->keyDim);
        //echo "valueDim=";var_dump($this->valueDim);
        //echo "query_shape=(".implode(',',$query_shape).")\n";   // ((Batch,) TargetSeq, Dim)
        //echo "value_shape=(".implode(',',$value_shape).")\n";   // ((Batch,) SourceSeq, Dim)
        //echo "key_shape=(".implode(',',$key_shape).")\n";       // ((Batch,) SourceSeq, Dim)

        $query_rank = count($query_shape);                      // rank = 2 (full_rank = 3)
        $value_rank = count($value_shape);                      // rank = 2 (full_rank = 3)
        $key_rank = count($key_shape);                          // rank = 2 (full_rank = 3)

        //echo "==query_dense==\n";
        [$einsum_equation, $bias_axes, $output_rank] = $this->build_proj_equation(
            $query_rank, bound_dims:1, output_dims:2
        );
        $common_args = $this->get_common_args();
        //echo "common_args=";
        //var_dump($common_args);
        echo "query_einsum=$einsum_equation\n";                 // ab.c,c.de->ab.de    =>  gemm(x,y)
        $this->query_dense = new EinsumDense(
            $this->backend,
            $einsum_equation,
            ...$common_args,
            output_shape:$this->get_output_shape(               // ((Batch), (TargetSeq), numHeads, keyDim)
                $output_rank,
                $query_shape,
                [$this->numHeads, $this->keyDim]
            ),
            bias_axes:($this->useBias)?$bias_axes:null,
            name:'query',
        );
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // query_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // query_dense/bias
            }
        }
        $this->query_dense->build($query_shape,sampleWeights:$sampleW);
        //echo "query_dense->inputShape=(".implode(',',$this->query_dense->inputShape()).")\n";
        //echo "query_dense->kernelShape=(".implode(',',$this->query_dense->getParams()[0]->shape()).")\n";
        //echo "query_dense->outputShape=(".implode(',',$this->query_dense->outputShape()).")\n";
        // input  = ((Batch,) TargetSeq, Dim)
        // kernel = (Dim, numHeads, keyDim)
        // output = ((Batch), (TargetSeq), numHeads, keyDim)

        //echo "==key_dense==\n";
        [$einsum_equation, $bias_axes, $output_rank] = $this->build_proj_equation(
            $key_rank, bound_dims:1, output_dims:2
        );
        echo "key_einsum=$einsum_equation\n";                   // ab.c,c.de->ab.de    =>  gemm(x,y)
        //echo "common_args=";
        //var_dump($common_args);
        $this->key_dense = new EinsumDense(
            $this->backend,
            $einsum_equation,
            ...$common_args,
            output_shape:$this->get_output_shape(               // ((Batch), (SourceSeq), numHeads, keyDim)
                $output_rank,
                $key_shape,
                [$this->numHeads, $this->keyDim]
            ),
            bias_axes:($this->useBias)?$bias_axes:null,
            name:'key',
        );
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // key_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // key_dense/bias
            }
        }
        $this->key_dense->build($key_shape,sampleWeights:$sampleW);
        //echo "key_dense->inputShape=(".implode(',',$this->key_dense->inputShape()).")\n";
        //echo "key_dense->kernelShape=(".implode(',',$this->key_dense->getParams()[0]->shape()).")\n";
        //echo "key_dense->outputShape=(".implode(',',$this->key_dense->outputShape()).")\n";
        // input  = ((Batch,) SourceSeq, Dim)
        // kernel = (Dim, numHeads, keyDim)
        // output = ((Batch), (SourceSeq), numHeads, keyDim)

        //echo "==value_dense==\n";
        [$einsum_equation, $bias_axes, $output_rank] = $this->build_proj_equation(
            $value_rank, bound_dims:1, output_dims:2
        );
        echo "value_einsum=$einsum_equation\n";                 // ab.c,c.de->ab.de    =>  gemm(x,y)
        //echo "common_args=";
        //var_dump($common_args);
        $this->value_dense = new EinsumDense(
            $this->backend,
            $einsum_equation,
            ...$common_args,
            output_shape:$this->get_output_shape(               // ((Batch), (SourceSeq), numHeads, valueDim)
                $output_rank,
                $value_shape,
                [$this->numHeads, $this->valueDim]
            ),
            bias_axes:($this->useBias)?$bias_axes:null,
            name:'value',
        );
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // value_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // value_dense/bias
            }
        }
        $this->value_dense->build($value_shape,sampleWeights:$sampleW);
        //echo "value_dense->inputShape=(".implode(',',$this->value_dense->inputShape()).")\n";
        //echo "value_dense->kernelShape=(".implode(',',$this->value_dense->getParams()[0]->shape()).")\n";
        //echo "value_dense->outputShape=(".implode(',',$this->value_dense->outputShape()).")\n";
        // input  = ((Batch,) SourceSeq, Dim)
        // kernel = (Dim, numHeads, valueDim)
        // output = ((Batch), (SourceSeq), numHeads, valueDim)

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports multi-head einsum computations.
        //echo "==build_attention==\n";
        $this->build_attention($output_rank);

        // scores = einsum(equation, key, query)
        // key:    ((Batch), (SourceSeq), numHeads, keyDim)
        // query:  ((Batch), (TargetSeq), numHeads, keyDim)
        // scores: ((Batch), numHeads, (TargetSeq), (SourceSeq))
        //echo "dot_product_equation=".$this->dot_product_equation."\n";  // aecd,abcd->acbe

        // output = einsum(equation,scores,value)
        // scores: ((Batch), numHeads, (TargetSeq), (SourceSeq))
        // value:  ((Batch), (SourceSeq), numHeads, valueDim)
        // output: ((Batch), (TargetSeq), numHeads, valueDim)
        //echo "combine_equation=".$this->combine_equation."\n";          // acbe,aecd->abcd

        //echo "==output_dense==\n";
        //echo "common_args=";
        //var_dump($common_args);
        $output_dense_input_shape = $this->query_dense->outputShape();

        // input:   ((Batch), (TargetSeq), numHeads, valueDim)
        // kernel:  (numHeads, valueDim, Dim)
        // output:  ((Batch), (TargetSeq), Dim)
        // equation: ab.cd,cd.e->ab.e    =>  gemm(x,y)
        $this->output_dense = $this->make_output_dense(
            $query_shape,
            $output_dense_input_shape,
            ...$common_args,
            name:'attention_output',
        );
        //echo "output_dense_input_shape0=(".implode(',',$output_dense_input_shape).")\n";
        $output_dense_input_shape[count($output_dense_input_shape)-1] = $this->valueDim;
        //echo "output_dense_input_shape1=(".implode(',',$output_dense_input_shape).")\n";
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // output_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // output_dense/bias
            }
        }
        $this->output_dense->build($output_dense_input_shape,sampleWeights:$sampleW);

        $this->outputShape = $this->output_dense->outputShape();
        //echo "output_dense->inputShape=(".implode(',',$this->output_dense->inputShape()).")\n";
        //echo "output_dense->kernelShape=(".implode(',',$this->output_dense->getParams()[0]->shape()).")\n";
        //echo "output_dense->outputShape=(".implode(',',$this->outputShape).")\n";

        // scores: ((Batch), numHeads, (TargetSeq), (SourceSeq))
        $targetSeq = $this->query_dense->outputShape()[0];
        $sourceSeq = $this->key_dense->outputShape()[0];
        $this->scoresShape =[$this->numHeads,$targetSeq,$sourceSeq];
        //$this->syncWeightVariables();
    }

    public function getParams() : array
    {
        return array_merge(
            $this->query_dense->getParams(),
            $this->key_dense->getParams(),
            $this->value_dense->getParams(),
            $this->output_dense->getParams(),
        );
    }

    public function getGrads() : array
    {
        return array_merge(
            $this->query_dense->getGrads(),
            $this->key_dense->getGrads(),
            $this->value_dense->getGrads(),
            $this->output_dense->getGrads(),
        );
    }

    public function getConfig() : array
    {
        return [
            'num_heads'=>$this->numHeads,
            'key_dim'=>$this->keyDim,
            'options' => [
                'value_dim'=>$this->valueDim,
                'dropout'=>$this->dropout,
                'use_bias'=>$this->useBias,
                'output_shape'=>$this->outputShape,
                'attention_axes'=>$this->attentionAxes,
                'kernel_initializer' => $this->kernelInitializerName,
                'bias_initializer' => $this->biasInitializerName,
            ],
        ];
    }

    /**
     * @param <string,mixed>
     */
    private function get_common_args() : array
    {
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        return [
            'kernel_initializer'=>$this->kernelInitializerName,
            'bias_initializer'=>$this->biasInitializerName,
        ];
    }

    /*
        """Builds the output projection matrix.

        Args:
            free_dims: Number of free dimensions for einsum equation building.
            common_kwargs: Common keyword arguments for einsum layer.
            name: Name for the projection layer.

        Returns:
            Projection layer.
        """
    */
    private function make_output_dense(
        array $query_shape,
        array $output_dense_input_shape,
        string $name=null,
        mixed ...$common_args,
        ) : Layer
    {
        $query_rank = count($query_shape);
        if($this->partial_output_shape) {
            $output_shape = $this->partial_output_shape;  // without batch and sequence

        } else {
            $output_shape = array_slice($query_shape,-1);  // without batch and sequence
        }
        [$einsum_equation, $bias_axes, $output_rank] = $this->build_proj_equation(
            $query_rank, bound_dims:2, output_dims:count($output_shape)
        );

        // input:   ((Batch), (TargetSeq), numHeads, valueDim)
        // kernel:  (numHeads, valueDim, Dim)
        // output:  ((Batch), (TargetSeq), Dim)

        echo "output_einsum=$einsum_equation\n";        //  abcd,cde->abe
        return new EinsumDense(
            $this->backend,
            $einsum_equation,
            ...$common_args,
            output_shape:$this->get_output_shape(       // output: ((Batch), (TargetSeq), Dim)
                $output_rank,
                $output_dense_input_shape,
                $output_shape
            ),
            bias_axes:($this->useBias)?$bias_axes:null,
            name:$name,
        );
    }

    /*
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
            rank: the rank of query, key, value tensors.
        """
    */
    private function build_attention($rank) : void
    {
        if($this->attention_axes===null) {
            $this->attention_axes = range(1, $rank-2);
        }
        //echo "attention rank=$rank\n";
        //echo "attention_axes=(".implode(',',$this->attention_axes).")\n";
        [
            $this->dot_product_equation,
            $this->backward_dot_product_key_equation,
            $this->backward_dot_product_query_equation,
            $this->combine_equation,
            $this->backward_combine_scores_equation,
            $this->backward_combine_value_equation,
            $attn_scores_rank,
        ] = $this->build_attention_equation($rank, attn_axes:$this->attention_axes);
        echo "dot_product_equation: ".$this->dot_product_equation."\n";
        echo "combine_equation: ".$this->combine_equation."\n";
        //echo "attn_scores_rank=$attn_scores_rank\n";
        $norm_axes = range(
            $attn_scores_rank - count($this->attention_axes), $attn_scores_rank-1
        );
        //$this->softmax = new Activation($this->backend,'softmax',axis:$norm_axes);
        //echo "norm_axes=".implode(',',$norm_axes)."\n";
        $this->softmax_layer = new Activation($this->backend,'softmax');
        $this->dropout_layer = new Dropout(
            $this->backend,
            rate:$this->dropout
        );
        $this->inverse_sqrt_key_dim = 1.0 / sqrt($this->keyDim);
    }

    private function masked_softmax(
        NDArray $attention_scores,
        NDArray $attention_mask=null,
        bool $training=null,
    ) : NDArray
    {
        $K = $this->backend;
        # Normalize the attention scores to probabilities.
        # attention_scores = [B, N, T, S]
        if ($attention_mask!==null) {
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            //echo "attention_scores=(".implode(',',$attention_scores->shape()).")\n";
            //echo "attention_mask=(".implode(',',$attention_mask->shape()).")\n";
            //$mask_expansion_axis = -count($this->attention_axes) * 2 - 1;
            //echo "mask_expansion_axis=".$mask_expansion_axis."\n";
            //$n = $attention_scores->ndim() - $attention_mask->ndim();
            //for($i=0;$i<$n;++$i) {
            //    $attention_mask = $K->expandDims(
            //        $attention_mask, axis:$mask_expansion_axis
            //    );
            //}
            //echo "expanded_attention_mask=(".implode(',',$attention_mask->shape()).")\n";

        }
        $results = $this->softmax_layer->_rawCall(
            [$attention_scores], ['training'=>$training, 'mask'=>$attention_mask]);
        return $results[0];
    }

    /*
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, S, N, key_dim)`.
            value: Projected value tensor of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
    */
    private function compute_attention(
        NDArray $query,
        NDArray $key,
        NDArray $value,
        NDArray $attention_mask=null,
        bool $training=null
    ) : array
    {
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.    
        $query = $K->scale(
            $this->inverse_sqrt_key_dim,
            $query,
        );

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        $attention_scores = $K->einsum($this->dot_product_equation, $key, $query);
    
        #echo "attention_scores: ".$mo->toString($attention_scores,format:'%10.7e',indent:true)."\n";
        $attention_scores = $this->masked_softmax(
            $attention_scores, $attention_mask, $training
        );
        #echo "softmax_attention_scores: ".$mo->toString($attention_scores,format:'%10.7e',indent:true)."\n";
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if($this->dropout!==null) {
            $final_attn_scores = $this->dropout_layer->_rawCall(
                [$attention_scores], ['training'=>$training]
            )[0];
        } else {
            $final_attn_scores = $attention_scores;
        }
    
        # `context_layer` = [B, T, N, H]
        $attention_output = $K->einsum(
            $this->combine_equation, $final_attn_scores, $value
        );
        return [$attention_output, $attention_scores];
    }

    private function compute_differntiate_attention(
        $dAttention_output,
        $query,
        $key,
        $value,
        $attention_output,
        $final_attn_scores,
        $attention_mask,
        $training,
    ) : array
    {
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();

        //echo "query: ".$mo->toString($query,indent:true)."\n";
        //echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,indent:true)."\n";

        $dValue = $K->einsum($this->backward_combine_value_equation, $dAttention_output, $final_attn_scores);
        $dScores = $K->einsum($this->backward_combine_scores_equation, $dAttention_output, $value);

        //echo "combine_equation: ".$this->combine_equation."\n";
        //echo "final_attn_scores=(".implode(',',$final_attn_scores->shape()).")\n";
        //echo "value=(".implode(',',$value->shape()).")\n";
        //echo "attention_output=(".implode(',',$attention_output->shape()).")\n";
        //echo "dAttention_output=(".implode(',',$dAttention_output->shape()).")\n";
        
        //echo "combine_dScore_equation: ".$this->backward_combine_scores_equation."\n";
        //echo "value=(".implode(',',$value->shape()).")\n";
        //echo "dOutput=(".implode(',',$dAttention_output->shape()).")\n";
        //echo "dSoftmax_scores=(".implode(',',$dScores->shape()).")\n";
        //echo ": ".$mo->toString($dScores,indent:true)."\n";

        //$dScores = $K->dSoftmax($dScores, $final_attn_scores);
        $dScores = $this->softmax_layer->_rawDifferentiate([$dScores])[0];

        //echo "dScores=(".implode(',',$dScores->shape()).")";
        //echo ": ".$mo->toString($dScores,format:'%10.7e',indent:true)."\n";

        //echo "dot_product_equation: ".$this->dot_product_equation."\n";
        //echo "backward_dot_product_key_equation: ".$this->backward_dot_product_key_equation."\n";
        //echo "backward_dot_product_query_equation: ".$this->backward_dot_product_query_equation."\n";
        $dKey = $K->einsum($this->backward_dot_product_key_equation, $dScores, $query);
        $dQuery = $K->einsum($this->backward_dot_product_query_equation, $dScores, $key);

        //echo "inverse_sqrt_key_dim: ".$this->inverse_sqrt_key_dim."\n";
        $dQuery = $K->scale(
            $this->inverse_sqrt_key_dim,
            $dQuery,
        );

        return [$dQuery, $dKey, $dValue];
    }

    /**
     * @param array<Variable> $inputs
     * @param array<Variable> $mask
     * @return array<Variable>|Variable
     */
    public function forward(
        array $inputs, 
        Variable|bool $training=null, 
        Variable|bool $returnAttentionScores=null,
        array $masks=null,
        NDArray $attention_mask=null,
        Variable|bool $useCausalMask=null,
    )
    {
        //$outputs = null;
        if(!is_array($inputs)) {
            throw new InvalidArgumentException('inputs must be list of Variable');
        }
        [$inputs,$rawInputs]     = $this->packAndUnpackVariables($this->backend,$inputs);
        $options = [];
        [$training,$rawTraining] = $this->packAndUnpackVariable($this->backend,$training,unbackpropagatable:true);
        [$returnAttentionScores,$rawReturnAttentionScores] = $this->packAndUnpackVariable($this->backend,$returnAttentionScores,unbackpropagatable:true);
        [$useCausalMask,$rawUseCausalMask] = $this->packAndUnpackVariable($this->backend,$useCausalMask,unbackpropagatable:true);
        $options['training'] = $training;
        $options['returnAttentionScores'] = $returnAttentionScores;
        $options['useCausalMask'] = $useCausalMask;
        $rawMasks = null;
        if($masks) {
            if(count($masks)!=2) {
                throw new InvalidArgumentException('mask must be list of the two of masks as queryMask and valueMask');
            }
            [$masks,$rawMasks] = $this->packAndUnpackVariables($this->backend,$masks,unbackpropagatable:true);
            $options['queryMask'] = $masks[0];
            $options['valueMask'] = $masks[1];
        }
        if(!$this->built) {
            $this->build($inputs);
            $this->built = true;
        }
        $options = $this->cleanNullValue($options);
        
        $numOfOutputs = $this->numOfOutputs($options);
        $session = $this->preGradientProcessOnSession($inputs,$options);
        $session->begin();
        try {
            $this->assertInputShapes($rawInputs,'forward');
            $this->unbackpropagatables = null;
            $rawOutputs = $this->call(
                $rawInputs, 
                training:$rawTraining, 
                returnAttentionScores:$rawReturnAttentionScores,
                masks:$rawMasks,
                attention_mask:$attention_mask,
                useCausalMask:$rawUseCausalMask,
            );
            if($returnAttentionScores){
                $this->assertOutputShape($rawOutputs[0],'forward');
                $this->assertScoresShape($rawOutputs[1],'forward');
            } else {
                $this->assertOutputShape($rawOutputs,'forward');
            }
        } finally{
            $session->end();
        }
        if($numOfOutputs==1) {
            $rawOutputs = [$rawOutputs];
        }
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session,$inputs,
            $rawOutputs,$this->unbackpropagatables);
        if($numOfOutputs==1) {
            return $outputs[0];
        } else {
            return $outputs;
        }
    }
    
    protected function call( 
        array $inputs,
        bool $training=null,
        bool $returnAttentionScores=null,
        array $masks=null,
        NDArray $attention_mask=null,
        bool $useCausalMask=null,
    ) : NDArray|array
    {
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();

        $container = $this->container();
        $query = $inputs[0] ?? null;
        $value = $inputs[1] ?? null;
        $key = $inputs[2] ?? $value;
        if(count($inputs)==3) {
            $container->sameKey = false;
        } else {
            $container->sameKey = true;
        }
        $query_mask = $masks[0] ?? null;
        $value_mask = $masks[1] ?? null;
        $key_mask = $masks[2] ?? $value_mask;

        $attention_mask = $this->compute_attention_mask(
                $query,
                $value,
                query_mask:$query_mask,
                value_mask:$value_mask,
                key_mask:$key_mask,
                attention_mask:$attention_mask,
                useCausalMask:$useCausalMask,
        );
    
        echo "query equation: ". $this->query_dense->getEquation()."\n";
        echo "key equation: ". $this->key_dense->getEquation()."\n";
        echo "value equation: ". $this->value_dense->getEquation()."\n";
        echo "output equation: ". $this->output_dense->getEquation()."\n";
        //echo "query before dense: ".$mo->toString($query,indent:true)."\n";
        //echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,indent:true)."\n";

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        //echo "query: ".$mo->toString($query,indent:true)."\n";
        $query = $this->query_dense->_rawCall([$query],['training'=>$training])[0];
        //echo "query_: ".$mo->toString($query,format:'%14.7f',indent:true)."\n";
    
        # `key` = [B, S, N, H]
        $key = $this->key_dense->_rawCall([$key],['training'=>$training])[0];
    
        # `value` = [B, S, N, H]
        $value = $this->value_dense->_rawCall([$value],['training'=>$training])[0];
    
        //echo "query after dense: ".$mo->toString($query,indent:true)."\n";
        //echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,indent:true)."\n";

        [$attention_output, $attention_scores] = $this->compute_attention(
                $query, $key, $value, $attention_mask, $training
        );
        $container->attention_output = $attention_output;

        //echo "attention_output before dense: ".$mo->toString($attention_output,indent:true)."\n";
        $attention_output = $this->output_dense->_rawCall([$attention_output],['training'=>$training])[0];
        //echo "attention_output after dense: ".$mo->toString($attention_output,indent:true)."\n";

        $container->attention_mask = $attention_mask;
        $container->training = $training;
        $container->query = $query;
        $container->key = $key;
        $container->value = $value;
        $container->attention_scores = $attention_scores;

        //echo "query: ".$mo->toString($query,indent:true)."\n";
        //echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,indent:true)."\n";

        if($returnAttentionScores) {
            return [$attention_output, $attention_scores];
        }
        return $attention_output;
    }
    
    protected function differentiate(NDArray $dOutputs) : array
    {
        $mo = $this->backend->localMatrixOperator();
        //echo "dOutputs: ".$mo->toString($dOutputs,indent:true)."\n";
        $container = $this->container();

        $dAttention_output = $this->output_dense->_rawDifferentiate([$dOutputs])[0];
        //echo "dAttention_output: ".$mo->toString($dAttention_output,indent:true)."\n";

        [$dQuery, $dKey, $dValue] = $this->compute_differntiate_attention(
            $dAttention_output,
            $container->query,
            $container->key,
            $container->value,
            $container->attention_output,
            $container->attention_scores,
            $container->attention_mask,
            $container->training
        );

        //echo "============================================\n";
        //echo "dQuery before dense->df(): ".$mo->toString($dQuery,indent:true)."\n";

        $dValue = $this->value_dense->_rawDifferentiate([$dValue])[0];
        $dQuery = $this->query_dense->_rawDifferentiate([$dQuery])[0];
        $dKey   = $this->key_dense->_rawDifferentiate([$dKey])[0];

        //echo "============================================\n";
        //echo "dQuery after dense->df(): ".$mo->toString($dQuery,indent:true)."\n";

        $results = [$dQuery, $dValue];
        if(!$container->sameKey) {
            $results[] = $dKey;
        }
        return $results;
    }

    /*
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `useCausalMask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, T, N, key_dim)`.
            value: Projected value tensor of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            useCausalMask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `useCausalMask=True`.
        """
    */
    private function compute_attention_mask(
        NDArray $query,
        NDArray $value,
        NDArray $query_mask=null,
        NDArray $value_mask=null,
        NDArray $key_mask=null,
        NDArray $attention_mask=null,
        bool $useCausalMask=null,
    ) : ?NDArray
    {
        $K = $this->backend;
        $auto_mask = null;
        //if($query_mask) {
        //    $query_mask = $K->cast($query_mask, NDArray::bool);
        //    # B = batch size, T = max query length
        //    $auto_mask = $K->expand_dims($query_mask, -1);  # shape is [B, T, 1]
        //}
        //if($value_mask) {
        //    $value_mask = $K->cast($value_mask, NDArray::bool);
        //    # B = batch size, S == max value length
        //    $mask = $K->expand_dims($value_mask, -2);  # shape is [B, 1, S]
        //    $auto_mask = ($auto_mask===null) ? $mask : ($auto_mask & $mask);
        //}
        //if($key_mask) {
        //    $key_mask = $K->cast($key_mask, NDArray::bool);
        //    # B == batch size, S == max key length == max value length
        //    $mask = $K->expand_dims($key_mask, -2);  # shape is [B, 1, S]
        //    $auto_mask = ($auto_mask===null) ? $mask : ($auto_mask & $mask);
        //}
        if($useCausalMask) {
            #original <the shape of the causal mask is [1, T, S]>
            #rindow<the shape of the causal mask is [T, S]>
            $mask = $this->compute_causal_mask($query, $value);
            echo "causal-mask".$K->localMatrixOperator()->shapeToString($mask->shape()).": ".$K->localMatrixOperator()->toString($mask,indent:true)."\n";
            $auto_mask = ($auto_mask===null) ? $mask : ($auto_mask & $mask);
        }
        if($auto_mask) {
            # merge attention_mask & automatic mask, to shape [B, T, S]
            //$attention_mask = [
            //    ($attention_mask===null) ? $auto_mask : ($K->cast($attention_mask, NDArray::bool) & $auto_mask)
            //];
            $attention_mask = $auto_mask;
        }
        return $attention_mask;
    }
    
    /*
    """Computes a causal mask (e.g., for masked self-attention layers).
    
    For example, if query and value both contain sequences of length 4,
    this function returns a boolean tensor equal to:

    ```
    [[[True,  False, False, False],
      [True,  True,  False, False],
      [True,  True,  True,  False],
      [True,  True,  True,  True]]]
    ```

    Args:
        query: query tensor of shape `(B, T, ...)`.
        value: value tensor of shape `(B, S, ...)` (optional, defaults to
            query).

    Returns:
        mask: a boolean tensor of shape `(1, T, S)` containing a lower
            triangular matrix of shape `(T, S)`.
    """
    */
    private function compute_causal_mask(
        NDArray $query,
        NDArray $value=null
    ) : NDArray
    {
        $K = $this->backend;
        $q_seq_length = $query->shape()[1];
        $v_seq_length = ($value===null) ? $q_seq_length : $value->shape()[1];
        $ones_mask = $K->ones([$q_seq_length, $v_seq_length],dtype:NDArray::float32);
        $row_index = $K->cumsum($ones_mask, axis:-2);
        $col_index = $K->cumsum($ones_mask, axis:-1);
        $mask = $K->sub($row_index, $col_index);
        return $K->greaterEqual($mask, 0);
    }
    
    private function compute_output_shape(
        array $query_shape,
        array $value_shape,
        array $key_shape=null,
    ) : array
    {
        if($key_shape===null) {
            $key_shape = $value_shape;
        }
    
        if($query_shape[-1] != $value_shape[-1]) {
            throw new ValueError(
                "The last dimension of `query_shape` and `value_shape` ".
                "must be equal, but are {query_shape[-1]}, {value_shape[-1]}. ".
                "Received: query_shape={query_shape}, value_shape={value_shape}"
            );
        }
    
        if($value_shape[R(1,-1)] != $key_shape[R(1,-1)]) {
            throw new ValueError(
                "All dimensions of `value` and `key`, except the last one, ".
                "must be equal. Received: value_shape={value_shape} and ".
                "key_shape={key_shape}"
            );
        }
    
        if($this->partial_output_shape) {
            return $query_shape[R(null,-1)] + $this->partial_output_shape;
        }
    
        return $query_shape;
    }
    
    private function compute_output_spec(
        NDArray $query,
        NDArray $value,
        NDArray $key=null,
        NDArray $query_mask=null,
        NDArray $value_mask=null,
        NDArray $key_mask=null,
        NDArray $attention_mask=null,
        bool $returnAttentionScores=null,
        bool $training=null,
        bool $useCausalMask=null,
    ) : array
    {
        if($key) {
            $key_shape = $key->shape();
        } else {
            $key_shape = null;
        }
        $output_shape = $this->compute_output_shape(
            $query->shape(), $value->shape(), $key_shape
        );
        $output_spec = $K->KerasTensor(
            $output_shape, dtype:$this->compute_dtype
        );
        if($returnAttentionScores) {
            $length = $query->shape()[1];
            $attention_shape = [$query->shape()[0], $this->numHeads, $length, $length];
            return [$output_spec, $K->KerasTensor(
                $attention_shape, dtype:$this->compute_dtype
            )];
        }
        return $output_spec;
    }
    

    /*
        """Coverts an index to a einsum variable name.
    
        We simply map indices to lowercase characters, e.g. 0 -> 'a', 1 -> 'b'.
        """
    */
    private function index_to_einsum_variable(int $i) : string
    {
        return chr(ord('a')+$i);
    }

    /*
    */
    private function generate_equations(
        string $input_a_notation,
        string $input_b_notation,
        string $output_notation,
    ) : array
    {
        return [
            "{$input_a_notation},{$input_b_notation}->{$output_notation}",
            "{$output_notation},{$input_b_notation}->{$input_a_notation}",
            "{$output_notation},{$input_a_notation}->{$input_b_notation}",
        ];
    }
    /*
        """Builds einsum equations for the attention computation.
    
        Query, key, value inputs after projection are expected to have the shape as:
        `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
        `bs` and `<non-attention dims>` are treated as `<batch dims>`.
    
        The attention operations can be generalized:
        1. Query-key dot product:
            (<batch dims>, <query attention dims>, num_heads, channels),
            (<batch dims>, <key attention dims>, num_heads, channels) ->
            (<batch dims>, num_heads, <query attention dims>, <key attention dims>)
        2. Combination:
            (<batch dims>, num_heads, <query attention dims>, <key attention dims>),
            (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
            dims>, <query attention dims>, num_heads, channels)
    
        Args:
            rank: Rank of query, key, value tensors.
            attn_axes: List/tuple of axes, `[-1, rank)`,
                that attention will be applied to.
    
        Returns:
            Einsum equations.
        """
    */
    private function build_attention_equation(
        int $rank,
        array $attn_axes
    ) : array
    {
        $target_notation = "";
        $full_rank = $rank+1;
        for($i=0; $i<$full_rank; ++$i) {
            $target_notation .= $this->index_to_einsum_variable($i);
        }
        //echo "target_notation=$target_notation\n";
        # `batch_dims` includes the head dim.
        $batch_dims = range(0,$rank);
        //echo "batch_dims=(".implode(',',$batch_dims).")\n";
        $delete = array_merge($attn_axes, [$rank]);
        //echo "delete=[".implode(',',$delete)."]\n";
        foreach($delete as $i) {
            unset($batch_dims[$i]);
        }
        //echo "deleted batch_dims=(".implode(',',$batch_dims).")\n";
        $letter_offset = $full_rank;
        $source_notation = "";
        for($i=0;$i<$full_rank;++$i) {
            if(in_array($i,$batch_dims) || $i == $rank) {
                $source_notation .= $target_notation[$i];
            } else {
                $source_notation .= $this->index_to_einsum_variable($letter_offset);
                $letter_offset++;
            }
        }
        //echo "source_notation=$source_notation\n";
    
        $product_notation = implode('',array_merge(
            array_map(fn($i)=>$target_notation[$i],$batch_dims),
            array_map(fn($i)=>$target_notation[$i],$attn_axes),
            array_map(fn($i)=>$source_notation[$i],$attn_axes)
        ));
        [
            $dot_product_equation,
            $backward_dot_product_key_equation,
            $backward_dot_product_query_equation,
        ] = $this->generate_equations(
            $source_notation,  // key
            $target_notation,  // query
            $product_notation, // scores
        );
        $attn_scores_rank = strlen($product_notation)-1;
        [
            $combine_equation,
            $backward_combine_scores_equation,
            $backward_combine_value_equation,
        ] = $this->generate_equations(
            $product_notation,// scores
            $source_notation, // value
            $target_notation, // output
        );
        //echo "=========================================\n";
        //var_dump($backward_combine_scores_equation);
        return [
            $dot_product_equation,
            $backward_dot_product_key_equation,
            $backward_dot_product_query_equation,
            $combine_equation,
            $backward_combine_scores_equation,
            $backward_combine_value_equation,
            $attn_scores_rank,
        ];
    }

    /*
        """Builds an einsum equation for projections inside multi-head attention."""
    */
    private function build_proj_equation(
        int $free_dims,
        int $bound_dims,
        int $output_dims
        ) : array
    {
        //echo "free_dims=$free_dims\n";
        //echo "bound_dims=$bound_dims\n";
        //echo "output_dims=$output_dims\n";
        $input_str = "";
        $kernel_str = "";
        $output_str = "";
        $bias_axes = "";
        $letter_offset = 0;
        for($i=0;$i<$free_dims;++$i) {
            $char = $this->index_to_einsum_variable($i + $letter_offset);
            $input_str .= $char;
            $output_str .= $char;
        }
    
        $letter_offset += $free_dims;
        for($i=0;$i<$bound_dims;++$i) {
            $char = $this->index_to_einsum_variable($i + $letter_offset);
            $input_str .= $char;
            $kernel_str .= $char;
        }
    
        $letter_offset += $bound_dims;
        for($i=0;$i<$output_dims;++$i) {
            $char = $this->index_to_einsum_variable($i + $letter_offset);
            $kernel_str .= $char;
            $output_str .= $char;
            $bias_axes .= $char;
        }

        $equation = "{$input_str},{$kernel_str}->{$output_str}";
    
        //echo "build_proj_equation=[";
        //echo "equation:$equation, bias_axes:$bias_axes, output_rank:".(strlen($output_str)-1)."]\n";
        return [$equation, $bias_axes, strlen($output_str)-1];
    }
    
    private function get_output_shape(
        int $output_rank,
        array $input_shape,
        array $known_last_dims
        ) : array
    {
        $output_shape = array_merge(
            array_slice($input_shape,0,($output_rank - count($known_last_dims))),
            $known_last_dims
        );
        echo "output_shape=(".implode(',',$output_shape).")\n";
        return $output_shape;
    }
}
