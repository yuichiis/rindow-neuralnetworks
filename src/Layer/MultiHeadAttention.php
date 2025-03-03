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
    protected array $query_feature_shape;
    protected array $key_feature_shape;
    protected array $value_feature_shape;
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
        $value_shape = $inputShapes[1];  // Value
        if(count($inputShapes)==3) {
            if($inputShapes[1]!=$inputShapes[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
            $key_shape = $inputShapes[2]; // Key;
        }
        $key_shape ??= $value_shape;

        array_unshift($query_shape,1);
        array_unshift($value_shape,1);
        array_unshift($key_shape,1);

        //echo "numHeads=";var_dump($this->numHeads);
        //echo "keyDim=";var_dump($this->keyDim);
        //echo "valueDim=";var_dump($this->valueDim);
        //echo "query_shape=(".implode(',',$query_shape).")\n";   // (Batch, (Tq), Dim)
        //echo "value_shape=(".implode(',',$value_shape).")\n";   // (Batch, (Tv), Dim)
        //echo "key_shape=(".implode(',',$key_shape).")\n";       // (Batch, (Tv), Dim)

        $query_rank = count($query_shape);                      // rank = 3+?
        $value_rank = count($value_shape);                      // rank = 3+?
        $key_rank = count($key_shape);                          // rank = 3+?

        $common_args = $this->get_common_args();
        //echo "common_args=";
        //var_dump($common_args);
        //echo "==query_dense==\n";
        //[$einsum_equation, $bias_axes, $output_rank] = $this->build_proj_equation(
        //    free_dims:$query_rank-1, bound_dims:1, output_dims:2
        //);
        //echo "query_einsum=$einsum_equation\n";
        //echo "output_rank=$output_rank\n";
        [$batch,$Tq,$Fq,$units,$dense_input_shape] = $this->build_dense_args(
            $query_shape,[$this->numHeads, $this->keyDim],
        );
        //echo "query_shape=".$this->shapeToString($query_shape)."\n";
        //echo "(B*Tq,Dim),(Dim,Head*KeyDim) -> (B*Tq,Head*KeyDim)\n"; // ab.c,c.de->ab.de =>
        //echo "B={$batch},Tq=".$this->shapeToString($Tq).",Dim={$dim},Head*KeyDim={$units}\n";//        m.n,n.k,=>m.k
        //                                          // gemm(batches.Dim,Dim.units) => batches.units
        $this->query_feature_shape = $Fq;           // kernel(Dim.units) , bias(units)
        $this->query_dense = new Dense(             // Dense(inputs(batches.Dim),units)
            $this->backend,                         //     units       : (numHeads.keyDim)
            $units,                                 //     input_shape : ((Tq),Dim)
            ...$common_args,                        //     output_shape: ((Tq),(numHeads.keyDim))
            input_shape:$dense_input_shape,         //     kernel_initializer
            name:'query_dense',                     //     bias_initializer
        );                                          //     use_bias
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // query_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // query_dense/bias
            }
        }
        $this->query_dense->build($dense_input_shape,sampleWeights:$sampleW);
        //echo "query_dense->inputShape=(".implode(',',$this->query_dense->inputShape()).")\n";
        //echo "query_dense->kernelShape=(".implode(',',$this->query_dense->getParams()[0]->shape()).")\n";
        //echo "query_dense->outputShape=(".implode(',',$this->query_dense->outputShape()).")\n";
        // input  = ((Batch,) Tq, Dim)
        // kernel = (Dim, numHeads, keyDim)
        // output = ((Batch), (Tq), numHeads, keyDim)

        //echo "==key_dense==\n";
        //[$einsum_equation, $bias_axes, $output_rank] = $this->build_proj_equation(
        //    $key_rank-1, bound_dims:1, output_dims:2
        //);
        //echo "key_einsum=$einsum_equation\n";
        //echo "common_args=";
        //var_dump($common_args);
        [$batch,$Tk,$Fk,$units,$dense_input_shape] = $this->build_dense_args(
            $key_shape,[$this->numHeads, $this->keyDim],
        );
        //echo "key_shape=".$this->shapeToString($key_shape)."\n";
        //echo "(B*Tv,Dim),(Dim,Head*KeyDim) -> (B*Tv,Head*KeyDim)\n";  // ab.c,c.de->ab.de =>
        //echo "B={$batch},Tv=".$this->shapeToString($Tv).",Dim={$dim},Head*KeyDim={$units}\n"; //        m.n,n.k,=>m.k
        //                                          // gemm(batches.Dim,Dim.units) => batches.units
        $this->key_feature_shape = $Fk;             // kernel(Dim.units) , bias(units)
        $this->key_dense = new Dense(               // Dense(inputs(batches.Dim),units)
            $this->backend,                         //     units       : (numHeads.keyDim)
            $units,                                 //     input_shape : ((Tq),Dim)
            ...$common_args,                        //     output_shape: ((Tq),(numHeads.keyDim))
            input_shape:$dense_input_shape,         //     kernel_initializer
            name:'key_dense',                       //     bias_initializer
        );                                          //     use_bias
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // key_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // key_dense/bias
            }
        }
        $this->key_dense->build($dense_input_shape,sampleWeights:$sampleW);
        //echo "key_dense->inputShape=(".implode(',',$this->key_dense->inputShape()).")\n";
        //echo "key_dense->kernelShape=(".implode(',',$this->key_dense->getParams()[0]->shape()).")\n";
        //echo "key_dense->outputShape=(".implode(',',$this->key_dense->outputShape()).")\n";
        // input  = ((Batch,) Tv, Dim)
        // kernel = (Dim, numHeads, keyDim)
        // output = ((Batch), (Tv), numHeads, keyDim)

        //echo "==value_dense==\n";
        //[$einsum_equation, $bias_axes, $output_rank] = $this->build_proj_equation(
        //    $value_rank-1, bound_dims:1, output_dims:2
        //);
        //echo "value_einsum=$einsum_equation\n";
        //echo "common_args=";
        //var_dump($common_args);
        [$batch,$Tv,$Fv,$units,$dense_input_shape] = $this->build_dense_args(
            $value_shape,[$this->numHeads, $this->valueDim],
        );
        //echo "value_shape=".$this->shapeToString($value_shape)."\n";
        //echo "(B*Tv,Dim),(Dim,Head*KeyDim) -> (B*Tv,Head*KeyDim)\n";  // ab.c,c.de->ab.de => ab.de
        //echo "B={$batch},Tv=".$this->shapeToString($Tv).",Dim={$dim},Head*KeyDim={$units}\n"; //        m.n,n.k,=>m.k
        //                                          // gemm(batches.Dim,Dim.units) => batches.units
        $this->value_feature_shape = $Fv;           // kernel(Dim.units) , bias(units)
        $this->value_dense = new Dense(             // Dense(inputs(batches.Dim),units)
            $this->backend,                         //     units       : (numHeads.keyDim)
            $units,                                 //     input_shape : ((Tq),Dim)
            ...$common_args,                        //     output_shape: ((Tq),(numHeads.keyDim))
            input_shape:$dense_input_shape,         //     kernel_initializer
            name:'value_dense',                     //     bias_initializer
        );                                          //     use_bias
        $output_rank = 1+count($Tv)+1+1;
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // value_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // value_dense/bias
            }
        }
        $this->value_dense->build($dense_input_shape,sampleWeights:$sampleW);
        //echo "value_dense->inputShape=(".implode(',',$this->value_dense->inputShape()).")\n";
        //echo "value_dense->kernelShape=(".implode(',',$this->value_dense->getParams()[0]->shape()).")\n";
        //echo "value_dense->outputShape=(".implode(',',$this->value_dense->outputShape()).")\n";
        // input  = ((Batch,) Tv, Dim)
        // kernel = (Dim, numHeads, valueDim)
        // output = ((Batch), (Tv), numHeads, valueDim)

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports multi-head einsum computations.
        //echo "==build_attention==\n";
        $this->build_attention($output_rank);

        // scores = einsum(equation, key, query)
        // key:    ((Batch), (Tv), numHeads, keyDim)
        // query:  ((Batch), (Tq), numHeads, keyDim)
        // scores: ((Batch), numHeads, (Tq), (Tv))
        //echo "dot_product_equation=".$this->dot_product_equation."\n";  // aecd,abcd->acbe

        // output = einsum(equation,scores,value)
        // scores: ((Batch), numHeads, (Tq), (Tv))
        // value:  ((Batch), (Tv), numHeads, valueDim)
        // output: ((Batch), (Tq), numHeads, valueDim)
        //echo "combine_equation=".$this->combine_equation."\n";          // acbe,aecd->abcd

        //echo "==output_dense==\n";
        //echo "common_args=";
        //var_dump($common_args);
        //$output_dense_input_shape = $this->query_dense->outputShape();
        //array_unshift($output_dense_input_shape,1);
        $output_dense_input_shape = array_merge([$batch],$Tq,[$this->numHeads, $this->valueDim]);

        // input:   ((Batch, (Tq)), (numHeads, valueDim))
        // kernel:  ((numHeads, valueDim), Fq)
        // output:  ((Batch, (Tq)), Fq)
        // equation: ab.cd,cd.e->ab.e    =>  gemm(x,y)
        [$batch,$To,$Fo,$units,$dense_input_shape] = $this->build_dense_args(
            $output_dense_input_shape, $Fq,
        );
        $this->output_dense = new Dense(
            $this->backend,
            $units,
            ...$common_args,
            input_shape:$dense_input_shape, 
            name:'attention_output',
        );

        //echo "output_dense_input_shape0=(".implode(',',$output_dense_input_shape).")\n";
        //echo "valueDim=".$this->valueDim."\n";
        //$output_dense_input_shape[count($output_dense_input_shape)-1] = $this->valueDim;
        //echo "output_dense_input_shape1=(".implode(',',$output_dense_input_shape).")\n";
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // output_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // output_dense/bias
            }
        }
        $this->output_dense->build($dense_input_shape,sampleWeights:$sampleW);

        $this->outputShape = array_merge($Tq,$Fq);
        //echo "output_dense->inputShape=(".implode(',',$this->output_dense->inputShape()).")\n";
        //echo "output_dense->kernelShape=(".implode(',',$this->output_dense->getParams()[0]->shape()).")\n";
        //echo "output_dense->outputShape=(".implode(',',$this->outputShape).")\n";

        // scores: ((Batch), numHeads, (Tq), (Tv))
        $n_attn_axes = count($this->attention_axes);
        $querySeq = array_slice($this->inputShape[0],0,$n_attn_axes);
        $keySeq = array_slice($this->inputShape[1],0,$n_attn_axes);
        $this->scoresShape =array_merge([$this->numHeads],$querySeq,$keySeq);
        //echo "scoresShape=(".implode(',',$this->scoresShape).")\n";
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
            'use_bias'=>$this->useBias,
        ];
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
    private function build_attention(
        int $rank   // full rank
        ) : void
    {
        if($this->attention_axes===null) {
            $this->attention_axes = range(1, $rank-2-1);
        }
        //echo "value_dense output rank=$rank\n";
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
        //echo "dot_product_equation: ".$this->dot_product_equation."\n";
        //echo "attn_scores_rank=$attn_scores_rank\n";
        //echo "combine_equation: ".$this->combine_equation."\n";
        $norm_axes = range(
            $attn_scores_rank-count($this->attention_axes), $attn_scores_rank-1
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
        NDArray $mask=null,
        NDArray $attention_mask=null,
        NDArray $causal_mask=null,
        bool $training=null,
    ) : NDArray
    {
        $K = $this->backend;
        # Normalize the attention scores to probabilities.
        # attention_scores = [B, N, T, S]
        //if ($causal_mask!==null) {
        //    # The expand dim happens starting from the `num_heads` dimension,
        //    # (<batch_dims>, num_heads, <query_attention_dims,
        //    # key_attention_dims>)
        //    # mask_expansion_axis = -len(self._attention_axes) * 2 - 1
        //    # for _ in range(
        //    #     len(attention_scores.shape) - len(attention_mask.shape)
        //    # ):
        //    #     attention_mask = ops.expand_dims(
        //    #         attention_mask, axis=mask_expansion_axis
        //    #     )
        //    # return self._softmax(attention_scores, mask=attention_mask)
        //    echo "attention_scores=(".implode(',',$attention_scores->shape()).")\n";
        //    echo "causal_mask=(".implode(',',$causal_mask->shape()).")\n";
        //    //echo "attention_mask".$K->localMatrixOperator()->shapeToString($attention_mask->shape()).": ".$K->localMatrixOperator()->toString($attention_mask,indent:true)."\n";
        //    //$mask_expansion_axis = -count($this->attention_axes) * 2 - 1;
        //    //echo "mask_expansion_axis=".$mask_expansion_axis."\n";
        //    //$n = $attention_scores->ndim() - $attention_mask->ndim();
        //    //for($i=0;$i<$n;++$i) {
        //    //    $attention_mask = $K->expandDims(
        //    //        $attention_mask, axis:$mask_expansion_axis
        //    //    );
        //    //}
        //    //echo "expanded_attention_mask=(".implode(',',$attention_mask->shape()).")\n";
        //    $attention_scores = $K->masking($causal_mask,$attention_scores,fill:-1e9,axis:-$attention_mask->ndim());
        //}
        $original_shape  = $attention_scores->shape();
        $value_seq_shape = $attention_scores->shape();
        $shape = array_splice($value_seq_shape,0,-count($this->attention_axes));
        $shape = array_merge($shape,[(int)array_product($value_seq_shape)]);
        $attention_scores = $attention_scores->reshape($shape);
        $results = $K->softmax($attention_scores);
        $results = $results->reshape($original_shape);
        return $results;
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
        NDArray $query_mask=null,
        NDArray $value_mask=null,
        NDArray $key_mask=null,
        NDArray $attention_mask=null,
        bool $useCausalMask=null,
        bool $training=null
    ) : array
    {
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();

        //echo "query_           ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($query,axis:-1),axis:-1),
        //    $mo->array([
        //        [  8.000001,  21.333334,  34.666668,  48.000004,  61.333332,  74.66667 ],
        //        [ 87.99999 , 101.333336, 114.666664, 127.99999 , 141.33333 , 154.66666 ],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "value_           ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($value,axis:-1),axis:-1),
        //    $mo->array([
        //      [  6.8571434,  18.285715,   29.714287,   41.142857,   52.571426,   64.0,  75.42857  ],
        //      [ 86.85714  ,  98.28571 ,  109.714294,  121.14285 ,  132.57143 ,  144.0, 155.42856  ],                
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "key_             ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($key,axis:-1),axis:-1),
        //    $mo->array([
        //      [  6.8571434,  18.285715,   29.714287,   41.142857,   52.571426,   64.0,   75.42857  ],
        //      [ 86.85714  ,  98.28571 ,  109.714294,  121.14285 ,  132.57143 ,  144.0,  155.42856  ],                
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}

        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.    
        $scaled_query = $K->scale(
            $this->inverse_sqrt_key_dim,
            $query,
        );
        //echo "scaled_query     ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($scaled_query,axis:-1),axis:-1),
        //    $mo->array([
        //        [ 4.0000005, 10.666667,  17.333334,  24.000002,  30.666666,  37.333336 ],
        //        [43.999996 , 50.666668,  57.333332,  63.999996,  70.666664,  77.33333  ],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        //echo "dot_product:\n";
        //echo "  equation:".$this->dot_product_equation."\n";
        //echo "  key:    (".implode(',',$key->shape()).")\n";
        //echo "  query:  (".implode(',',$query->shape()).")\n";
        //echo "key_: ".$mo->toString($key,format:'%12.7f',indent:true)."\n";
        //echo "query_scaled: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        $attention_scores = $K->einsum($this->dot_product_equation, $key, $scaled_query);
        //echo "  scores: (".implode(',',$attention_scores->shape()).")\n";
        //echo "attention_scores: ".$mo->toString($attention_scores,format:'%12.7f',indent:true)."\n";
        //echo "attention_scores ";
        //if(!$mo->la()->isclose(
        //    $K->sum($attention_scores,axis:1),
        //    $mo->array([
        //    [[  0.857143 ,   2.2857149,   3.7142866,   5.142857,    6.5714297,  8.000001,    9.428573 ],
        //     [  2.2857149,   6.095238 ,   9.904762 ,  13.714286,   17.52381  , 21.333332,   25.142857 ],
        //     [  3.7142863,   9.904762 ,  16.09524  ,  22.285717,   28.476192 , 34.666668,   40.857147 ],
        //     [  5.142858 ,  13.714288 ,  22.285719 ,  30.857145,   39.428574 , 48.000004,   56.571438 ],
        //     [  6.5714293,  17.52381  ,  28.476192 ,  39.42857 ,   50.38095  , 61.333336,   72.28571  ],
        //     [  8.000001 ,  21.333336 ,  34.66667  ,  48.000004,   61.33334  , 74.66668 ,   88.00001  ],],
        //   
        //    [[119.42857,   135.14284,   150.85716,   166.57141,   182.28568,  197.99998,   213.71425  ],
        //     [137.5238 ,   155.61903,   173.71432,   191.80954,   209.90477,  228.0    ,   246.09523  ],
        //     [155.61903,   176.09523,   196.57143,   217.0476 ,   237.5238 ,  258.0    ,   278.47617  ],
        //     [173.71425,   196.57141,   219.42857,   242.28568,   265.14282,  287.99997,   310.8571   ],
        //     [191.80951,   217.0476 ,   242.28574,   267.5238 ,   292.76193,  318.0    ,   343.23807  ],
        //     [209.90475,   237.52379,   265.14285,   292.76187,   320.3809 ,  347.99997,   375.61896  ],],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        
        # attention_mask = [B, T, S]
        //echo "attention_scores=".$mo->toString($attention_scores,format:'%13.8f',indent:true)."\n";
        $attention_scores = $this->compute_masked_attention_scores(
            $attention_scores,
            $query,
            $value,
            query_mask:$query_mask,
            value_mask:$value_mask,
            key_mask:$key_mask,
            attention_mask:$attention_mask,
            useCausalMask:$useCausalMask,
        );
        //echo "masked_attention_scores=".$mo->toString($attention_scores,format:'%13.7e',indent:true)."\n";
        //echo "scores0: ".$mo->toString($K->slice($attention_scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        //echo "attention_mask".$K->localMatrixOperator()->shapeToString($attention_mask->shape()).": ".$K->localMatrixOperator()->toString($attention_mask,indent:true)."\n";
        //echo "attention_scores: ".$mo->toString($attention_scores,format:'%12.7f',indent:true)."\n";
        //echo "masked_scores    ";
        //if(!$mo->la()->isclose(
        //    $K->sum($attention_scores,axis:1),
        //    $mo->array([
        //      [[ 8.5714298e-01,  2.2857149e+00,  3.7142866e+00,  5.1428571e+00, -8.0000000e+09, -8.0000000e+09, -8.0000000e+09],
        //       [ 2.2857149e+00,  6.0952382e+00,  9.9047623e+00,  1.3714286e+01, -8.0000000e+09, -8.0000000e+09, -8.0000000e+09],
        //       [ 3.7142863e+00,  9.9047623e+00,  1.6095240e+01,  2.2285717e+01, -8.0000000e+09, -8.0000000e+09, -8.0000000e+09],
        //       [ 5.1428580e+00,  1.3714288e+01,  2.2285719e+01,  3.0857145e+01, -8.0000000e+09, -8.0000000e+09, -8.0000000e+09],
        //       [ 6.5714293e+00,  1.7523809e+01,  2.8476192e+01,  3.9428570e+01, -8.0000000e+09, -8.0000000e+09, -8.0000000e+09],
        //       [ 8.0000010e+00,  2.1333336e+01,  3.4666672e+01,  4.8000004e+01, -8.0000000e+09, -8.0000000e+09, -8.0000000e+09],],
        //     
        //      [[ 1.1942857e+02,  1.3514284e+02,  1.5085716e+02,  1.6657141e+02,  1.8228568e+02,  1.9799998e+02, -8.0000000e+09],
        //       [ 1.3752380e+02,  1.5561903e+02,  1.7371432e+02,  1.9180954e+02,  2.0990477e+02,  2.2800000e+02, -8.0000000e+09],
        //       [ 1.5561903e+02,  1.7609523e+02,  1.9657143e+02,  2.1704761e+02,  2.3752380e+02,  2.5800000e+02, -7.9999995e+09],
        //       [ 1.7371425e+02,  1.9657141e+02,  2.1942857e+02,  2.4228568e+02,  2.6514282e+02,  2.8799997e+02, -7.9999995e+09],
        //       [ 1.9180951e+02,  2.1704761e+02,  2.4228574e+02,  2.6752380e+02,  2.9276193e+02,  3.1800000e+02, -7.9999995e+09],
        //       [ 2.0990475e+02,  2.3752379e+02,  2.6514285e+02,  2.9276187e+02,  3.2038089e+02,  3.4799997e+02, -7.9999995e+09],],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}

        $attention_scores = $this->masked_softmax(
            $attention_scores,
            training:$training,
        );
        //echo "softmax_attention_scores: ".$mo->toString($attention_scores,format:'%13.8e',indent:true)."\n";
        //echo "softmaxed_scores ";
        //if(!$mo->la()->isclose(
        //    $K->sum($attention_scores,axis:1),
        //    $mo->array([
        //      [[1.4999763e+00, 1.7932341e+00, 2.1438265e+00, 2.5629623e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //       [8.5337543e-01, 1.3738747e+00, 2.2118413e+00, 3.5609090e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //       [4.4300479e-01, 9.6043861e-01, 2.0822403e+00, 4.5143166e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //       [2.1431160e-01, 6.2569302e-01, 1.8267403e+00, 5.3332543e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //       [9.8568387e-02, 3.8753179e-01, 1.5236220e+00, 5.9902773e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //       [4.3778196e-02, 2.3178329e-01, 1.2271748e+00, 6.4972639e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],],
        //     
        //      [[3.7331064e-04, 2.6616345e-03, 1.8977029e-02, 1.3530238e-01, 9.6468061e-01, 6.8780050e+00, 0.0000000e+00],
        //       [8.7833621e-05, 8.4332295e-04, 8.0970703e-03, 7.7742666e-02, 7.4643606e-01, 7.1667933e+00, 0.0000000e+00],
        //       [2.0426551e-05, 2.6410850e-04, 3.4148416e-03, 4.4152603e-02, 5.7087851e-01, 7.3812704e+00, 0.0000000e+00],
        //       [4.7118997e-06, 8.2042272e-05, 1.4284996e-03, 2.4872534e-02, 4.3307275e-01, 7.5405393e+00, 0.0000000e+00],
        //       [1.0806428e-06, 2.5338331e-05, 5.9412071e-04, 1.3930596e-02, 3.2663712e-01, 7.6588125e+00, 0.0000000e+00],
        //       [2.4681060e-07, 7.7931654e-06, 2.4607347e-04, 7.7698599e-03, 2.4533711e-01, 7.7466393e+00, 0.0000000e+00],],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if($this->dropout!=0) {
            //echo "dropout!!\n";
            $final_attn_scores = $this->dropout_layer->_rawCall(
                [$attention_scores], ['training'=>$training]
            )[0];
        } else {
            $final_attn_scores = $attention_scores;
        }
    
        # `context_layer` = [B, T, N, H]
        //echo "combine_product:\n";
        //echo "  equation:".$this->combine_equation."\n";
        //echo "  scores: (".implode(',',$final_attn_scores->shape()).")\n";
        //echo "  value:  (".implode(',',$value->shape()).")\n";
        $attention_output = $K->einsum(
            $this->combine_equation, $final_attn_scores, $value
        );
        //echo "  output: (".implode(',',$attention_output->shape()).")\n";
        //echo "outputs=".$mo->toString($attention_output,format:'%13.8f',indent:true)."\n";
        //echo "t_outputs        ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($attention_output,axis:-1),axis:-1),
        //    $mo->array([
        //        [ 26.528248,  30.400406,  33.525528,  35.827053,  37.436584,  38.539886],
        //        [142.1361  , 142.67139 , 143.04202 , 143.30365 , 143.49088 , 143.62624 ],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}

        return [$attention_output, $attention_scores, $scaled_query];
    }

    private function compute_differntiate_attention(
        $dAttention_output,
        $scaled_query,
        $key,
        $value,
        $attention_output,
        $softmaxed_attention_scores,
        $attention_mask,
        $training,
    ) : array
    {
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();

        //echo "d_outputs: ".$mo->toString($dAttention_output,indent:true)."\n";
        //echo "query_: ".$mo->toString($query,indent:true)."\n";
        //echo "key_: ".$mo->toString($key,indent:true)."\n";
        //echo "value_: ".$mo->toString($value,indent:true)."\n";
        //echo "scaledQuery      ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($scaled_query,axis:-1),axis:-1),
        //    $mo->array([
        //        [ 4.0000005, 10.666667,  17.333334,  24.000002,  30.666666,  37.333336 ],
        //        [43.999996 , 50.666668,  57.333332,  63.999996,  70.666664,  77.33333  ],                
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "value_           ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($value,axis:-1),axis:-1),
        //    $mo->array([
        //      [  6.8571434,  18.285715,   29.714287,   41.142857,   52.571426,   64.0, 75.42857 , ],
        //      [ 86.85714  ,  98.28571 ,  109.714294,  121.14285 ,  132.57143 ,  144.0, 155.42856, ],                
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "key_             ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($key,axis:-1),axis:-1),
        //    $mo->array([
        //      [  6.8571434,  18.285715,   29.714287,   41.142857,   52.571426,   64.0 ,   75.42857  ],
        //      [ 86.85714  ,  98.28571 ,  109.714294,  121.14285 ,  132.57143 ,  144.0 ,  155.42856  ],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "softmaxedScores  ";
        //if(!$mo->la()->isclose(
        //    $K->sum($softmaxed_attention_scores,axis:1),
        //    $mo->array([
        //        [[1.4999763e+00, 1.7932341e+00, 2.1438265e+00, 2.5629623e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //        [8.5337543e-01, 1.3738747e+00, 2.2118413e+00, 3.5609090e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //        [4.4300479e-01, 9.6043861e-01, 2.0822403e+00, 4.5143166e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //        [2.1431160e-01, 6.2569302e-01, 1.8267403e+00, 5.3332543e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //        [9.8568387e-02, 3.8753179e-01, 1.5236220e+00, 5.9902773e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //        [4.3778196e-02, 2.3178329e-01, 1.2271748e+00, 6.4972639e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],],
        //      
        //       [[3.7331064e-04, 2.6616345e-03, 1.8977029e-02, 1.3530238e-01, 9.6468061e-01, 6.8780050e+00, 0.0000000e+00],
        //        [8.7833621e-05, 8.4332295e-04, 8.0970703e-03, 7.7742666e-02, 7.4643606e-01, 7.1667933e+00, 0.0000000e+00],
        //        [2.0426551e-05, 2.6410850e-04, 3.4148416e-03, 4.4152603e-02, 5.7087851e-01, 7.3812704e+00, 0.0000000e+00],
        //        [4.7118997e-06, 8.2042272e-05, 1.4284996e-03, 2.4872534e-02, 4.3307275e-01, 7.5405393e+00, 0.0000000e+00],
        //        [1.0806428e-06, 2.5338331e-05, 5.9412071e-04, 1.3930596e-02, 3.2663712e-01, 7.6588125e+00, 0.0000000e+00],
        //        [2.4681060e-07, 7.7931654e-06, 2.4607347e-04, 7.7698599e-03, 2.4533711e-01, 7.7466393e+00, 0.0000000e+00],],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "dOutput          ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($dAttention_output,axis:-1),axis:-1),
        //    $mo->array([
        //        [ 320, 1120, 1920, 2720, 3520, 4320],
        //        [5120, 5920, 6720, 7520, 8320, 9120],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}




        $dValue = $K->einsum($this->backward_combine_value_equation, $dAttention_output, $softmaxed_attention_scores);
        $dSoftmaxedScores = $K->einsum($this->backward_combine_scores_equation, $dAttention_output, $value);

        //echo "combine_equation: ".$this->combine_equation."\n";
        //echo "d_outputs=(".implode(',',$dAttention_output->shape()).")\n";
        //echo ": ".$mo->toString($dAttention_output,indent:true)."\n";
        //echo "softmaxed_attention_scores=(".implode(',',$softmaxed_attention_scores->shape()).")\n";
        //echo ": ".$mo->toString($softmaxed_attention_scores,format:'%12.7e',indent:true)."\n";
        //echo "value=(".implode(',',$value->shape()).")\n";
        //echo "attention_output=(".implode(',',$attention_output->shape()).")\n";
        //echo "dAttention_output=(".implode(',',$dAttention_output->shape()).")\n";
        
        //echo "combine_dScore_equation: ".$this->backward_combine_scores_equation."\n";
        //echo "value=(".implode(',',$value->shape()).")\n";
        //echo "dOutput=(".implode(',',$dAttention_output->shape()).")\n";
        //echo "d_Value_=(".implode(',',$dValue->shape()).")\n";
        //echo ": ".$mo->toString($dValue,format:'%12.7e',indent:true)."\n";
        //echo "d_softmax_scores=(".implode(',',$dSoftmaxedScores->shape()).")\n";
        //echo ": ".$mo->toString($dSoftmaxedScores,format:'%12.7f',indent:true)."\n";
        //echo "dValue_          ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($dValue,axis:-1),axis:-1),
        //    $mo->array([
        //       [4.2566901e+02, 1.0029897e+03, 2.8493081e+03, 9.6420322e+03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        //       [3.2690841e-01, 2.6617122e+00, 2.3246796e+01, 2.2793692e+02, 2.6757715e+03, 3.9790055e+04, 0.0000000e+00],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "dSoftmaxedScores ";
        //if(!$mo->la()->isclose(
        //    $K->sum($dSoftmaxedScores,axis:1),
        //    $mo->array([
        //    [[   68.571434,   182.85716,    297.14285,    411.42853,    525.7143,    640.0,     754.2857  ],
        //     [  240.00002 ,   640.     ,   1040.     ,   1440.     ,   1839.9999,   2240.0,    2640.      ],
        //     [  411.42865 ,  1097.143  ,   1782.8573 ,   2468.5713 ,   3154.2854,   3840.0,    4525.7144  ],
        //     [  582.8572  ,  1554.2859 ,   2525.7146 ,   3497.1428 ,   4468.571 ,   5440.0,    6411.429   ],
        //     [  754.2857  ,  2011.4287 ,   3268.5718 ,   4525.7144 ,   5782.8564,   7040.0,    8297.144   ],
        //     [  925.71436 ,  2468.5715 ,   4011.4292 ,   5554.2856 ,   7097.1416,   8640.0,   10182.858   ],],
        //   
        //    [[13897.142   , 15725.713  ,  17554.287  ,  19382.857  ,  21211.428 ,  23040.0,   24868.57    ],
        //     [16068.571   , 18182.857  ,  20297.143  ,  22411.426  ,  24525.717 ,  26640.0,   28754.283   ],
        //     [18240.      , 20640.     ,  23040.002  ,  25440.     ,  27840.    ,  30240.0,   32639.998   ],
        //     [20411.428   , 23097.143  ,  25782.86   ,  28468.566  ,  31154.283 ,  33840.0,   36525.715   ],
        //     [22582.857   , 25554.283  ,  28525.717  ,  31497.143  ,  34468.566 ,  37440.0,   40411.426   ],
        //     [24754.283   , 28011.426  ,  31268.576  ,  34525.715  ,  37782.855 ,  41040.0,   44297.14    ],],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}


        $original_shape  = $softmaxed_attention_scores->shape();
        $value_seq_shape = $original_shape;
        $shape = array_splice($value_seq_shape,0,-count($this->attention_axes));
        $shape = array_merge($shape,[(int)array_product($value_seq_shape)]);
        $softmaxed_attention_scores = $softmaxed_attention_scores->reshape($shape);
        $dSoftmaxedScores = $dSoftmaxedScores->reshape($shape);
        //echo "d_softmax_scores=(".implode(',',$dSoftmaxedScores->shape()).")";
        //echo ": ".$mo->toString($dSoftmaxedScores,format:'%13.8f',indent:true)."\n";
        //echo "softmax_scores=(".implode(',',$softmaxed_attention_scores->shape()).")";
        //echo ": ".$mo->toString($softmaxed_attention_scores,format:'%13.8e',indent:true)."\n";
        $dScores = $K->dSoftmax($dSoftmaxedScores, $softmaxed_attention_scores);
        $dScores = $dScores->reshape($original_shape);
        //$dScores = $this->softmax_layer->_rawDifferentiate([$dScores])[0];

        //echo "attention_scores_flat=(".implode(',',$shape).")\n";
        //echo "dScores=(".implode(',',$dScores->shape()).")";
        //echo ": ".$mo->toString($dScores,format:'%12.7e',indent:true)."\n";
        //echo "query_=(".implode(',',$dScores->shape()).")";
        //echo ": ".$mo->toString($query,format:'%12.7e',indent:true)."\n";

        //echo "dScores: ".$mo->toString($K->sum($dScores,axis:1),format:'%13.7e',indent:true)."\n";
        //echo "dScores          ";
        //if(!$mo->la()->isclose(
        //    $K->sum($dScores,axis:1),
        //    $mo->array([
        //     [[-3.68827362e+01, -1.84759884e+01,  8.53789997e+00,  4.68208656e+01, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        //      [-8.78991776e+01, -7.28177872e+01, -6.63943911e+00,  1.67356430e+02, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        //      [-8.86066437e+01, -1.09776741e+02, -5.95193558e+01,  2.57902802e+02, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        //      [-6.59662399e+01, -1.16614609e+02, -1.18643356e+02,  3.01224579e+02, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        //      [-4.14447899e+01, -1.02046616e+02, -1.61780640e+02,  3.05272003e+02, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        //      [-2.34058514e+01, -7.92210159e+01, -1.82765610e+02,  2.85393036e+02, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],],
        //    
        //     [[-4.12724316e-01, -2.33427334e+00, -1.23053827e+01, -5.68087082e+01, -1.84536850e+02,  2.56400421e+02,  0.00000000e+00],
        //      [-1.13367267e-01, -8.65602612e-01, -6.17104435e+00, -3.87040062e+01, -1.74338898e+02,  2.20191666e+02,  0.00000000e+00],
        //      [-3.01261563e-02, -3.10288668e-01, -2.98748374e+00, -2.53812580e+01, -1.56907700e+02,  1.85616577e+02,  0.00000000e+00],
        //      [-7.81287625e-03, -1.08492844e-01, -1.40948188e+00, -1.61913509e+01, -1.36529984e+02,  1.54246017e+02,  0.00000000e+00],
        //      [-1.98902772e-03, -3.72262634e-02, -6.52189493e-01, -1.01179428e+01, -1.15917732e+02,  1.26725655e+02,  0.00000000e+00],
        //      [-4.99149377e-04, -1.25879552e-02, -2.97284395e-01, -6.22342253e+00, -9.66204147e+01,  1.03153618e+02,  0.00000000e+00],],
        //    ]),
        //    //atol:1e-3,
        //    rtol:1e-4,
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}

        //echo "dot_product_equation: ".$this->dot_product_equation."\n";
        //echo "backward_dot_product_key_equation: ".$this->backward_dot_product_key_equation."\n";
        //echo "backward_dot_product_query_equation: ".$this->backward_dot_product_query_equation."\n";
        $dKey = $K->einsum($this->backward_dot_product_key_equation, $dScores, $scaled_query);
        $dScaledQuery = $K->einsum($this->backward_dot_product_query_equation, $dScores, $key);
        //echo "equation=".$this->backward_dot_product_query_equation."\n";
        //echo "dScores=(".implode(',',$dScores->shape()).")";
        //echo ": ".$mo->toString($dScores,format:'%13.8e',indent:true)."\n";
        //echo "query_=(".implode(',',$key->shape()).")";
        //echo ": ".$mo->toString($scaled_query,format:'%12.7f',indent:true)."\n";
        //echo "key_=(".implode(',',$key->shape()).")";
        //echo ": ".$mo->toString($key,format:'%10.7f',indent:true)."\n";
        //echo "dKey_=(".implode(',',$dScaledQuery->shape()).")";
        //echo ": ".$mo->toString($dKey,format:'%12.7e',indent:true)."\n";
        //echo "dScaredQuery=(".implode(',',$dScaledQuery->shape()).")";
        //echo ": ".$mo->toString($dScaledQuery,format:'%10.7e',indent:true)."\n";
        //echo "dKey_            ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($dKey,axis:-1),axis:-1),
        //    $mo->array([
        //      [-7.93619080e+02, -1.45489856e+03, -1.96253760e+03,  4.21105859e+03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        //      [-3.28877878e+00, -2.18628464e+01, -1.48083786e+02, -1.01853784e+03, -6.29378076e+03,  7.48553125e+03,  0.00000000e+00],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "dScaredQuery: ".$mo->toString($K->sum($K->sum($dScaledQuery,axis:-1),axis:-1),format:'%12.7f',indent:true)."\n";

        //$dsq = $mo->array([
        //    [198.6606,  594.2465 , 778.41846, 785.38934, 700.2973 , 587.753  ],
        //    [495.0039,  391.81958, 311.45898, 248.00024, 197.50146, 157.14917],                
        //]);
        //echo "diff dScaredQuery: ".$mo->toString($K->sub(
        //    $K->sum($K->sum($dScaledQuery,axis:-1),axis:-1),
        //    $dsq,
        //),format:'%12.7f',indent:true)."\n";
        //echo "dScaredQuery     ";
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($dScaledQuery,axis:-1),axis:-1),
        //    $dsq,
        //    debug:true,
        //    //atol:1e-1,
        //    rtol:1e-3,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //if(!$mo->la()->isclose(
        //    $K->sum($K->sum($dKey,axis:-1),axis:-1),
        //    $mo->array([
        //      [-7.93619080e+02, -1.45489856e+03, -1.96253760e+03,  4.21105859e+03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        //      [-3.28877878e+00, -2.18628464e+01, -1.48083786e+02, -1.01853784e+03, -6.29378076e+03,  7.48553125e+03,  0.00000000e+00],                
        //    ]),
        //    //debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}

        //echo "inverse_sqrt_key_dim: ".$this->inverse_sqrt_key_dim."\n";
        $dQuery = $K->scale(
            $this->inverse_sqrt_key_dim,
            $dScaledQuery,
        );
        //echo "dQuery_=(".implode(',',$dQuery->shape()).")";
        //echo ": ".$mo->toString($dQuery,format:'%10.7e',indent:true)."\n";

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
        array $mask=null,
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
        $rawMask = null;
        if($mask) {
            if(count($mask)<2) {
                throw new InvalidArgumentException('mask must be list of 2 or 3 of masks as queryMask and valueMask and keyMask');
            }
            [$mask,$rawMask] = $this->packAndUnpackVariables($this->backend,$mask,unbackpropagatable:true);
            $options['queryMask'] = $mask[0] ?? null;
            $options['valueMask'] = $mask[1] ?? null;
            $options['keyMask']   = $mask[2] ?? null;
        } else {
            $rawMask = $this->retrieveMultiMasks($rawInputs);
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
                mask:$rawMask,
                attention_mask:$attention_mask,
                useCausalMask:$rawUseCausalMask,
            );
            if($returnAttentionScores){
                $this->assertOutputShape($rawOutputs[0],'forward');
                $this->assertScoresShape($rawOutputs[1],'forward');
                $rawOutputs[0] = $this->makeSingleMaskedValue($rawInputs[0], $rawOutputs[0]);
            } else {
                $this->assertOutputShape($rawOutputs,'forward');
                $rawOutputs = $this->makeSingleMaskedValue($rawInputs[0], $rawOutputs);
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
        array $mask=null,
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
        $query_mask = $mask[0] ?? null;
        $value_mask = $mask[1] ?? null;
        $key_mask   = $mask[2] ?? null;

        //echo "==========================================================\n";
        //echo "call(\n";
        //echo "  query: (".implode(',',$query->shape()).")\n";
        //echo "  value: (".implode(',',$value->shape()).")\n";
        //echo "  key  : (".implode(',',$key->shape()).")\n";
        //echo "  attention_mask  :(".(($attention_mask==null)?
        //    'null':implode(',',$attention_mask->shape())).")\n";
        //echo "  mask  : [".
        //    "(".(isset($mask[0])?implode(',',$mask[0]->shape()):'null')."),".
        //    "(".(isset($mask[1])?implode(',',$mask[1]->shape()):'null').")".
        //    "]\n";
        //echo "  useCausalMask:".($useCausalMask?'true':'false')."\n";
        //echo ")\n";
        //echo "query equation: ". $this->query_dense->getEquation()."\n";
        //echo "value equation: ". $this->value_dense->getEquation()."\n";
        //echo "key equation:   ". $this->key_dense->getEquation()."\n";
        //echo "dot_product equation: ". $this->dot_product_equation."\n";
        //echo "combine_equation:     ". $this->combine_equation."\n";
        //echo "output equation:      ". $this->output_dense->getEquation()."\n";


        //echo "query            ";
        //if(!$mo->la()->isclose(
        //    $query,
        //    $mo->array([
        //       [[0.01666667, 0.03333334, 0.05      , 0.06666667, 0.08333334],
        //        [0.1       , 0.11666667, 0.13333334, 0.15      , 0.16666667],
        //        [0.18333334, 0.2       , 0.21666667, 0.23333333, 0.25      ],
        //        [0.26666668, 0.28333333, 0.3       , 0.31666666, 0.33333334],
        //        [0.35      , 0.36666667, 0.38333333, 0.4       , 0.41666666],
        //        [0.43333334, 0.45      , 0.46666667, 0.48333332, 0.5       ]],
        //
        //       [[0.51666665, 0.53333336, 0.55      , 0.56666666, 0.5833333 ],
        //        [0.6       , 0.6166667 , 0.6333333 , 0.65      , 0.6666667 ],
        //        [0.68333334, 0.7       , 0.71666664, 0.73333335, 0.75      ],
        //        [0.76666665, 0.78333336, 0.8       , 0.81666666, 0.8333333 ],
        //        [0.85      , 0.8666667 , 0.8833333 , 0.9       , 0.9166667 ],
        //        [0.93333334, 0.95      , 0.96666664, 0.98333335, 1.        ]],
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}
        //echo "query before dense: ".$mo->toString($query,indent:true)."\n";
        //echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,indent:true)."\n";
        //echo "attention_mask".$K->localMatrixOperator()->shapeToString($attention_mask->shape()).": ".$K->localMatrixOperator()->toString($attention_mask,indent:true)."\n";

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        //echo "query: ".$mo->toString($query,indent:true)."\n";
        //echo "query: ".$mo->shapeToString($query->shape())."\n";
        [$full_input_shape,$full_output_shape,$Fq] = $this->make_forward_dense_shape(
            $query, [$this->numHeads, $this->keyDim],
        );
        $query = $query->reshape($full_input_shape);
        $query = $this->query_dense->_rawCall([$query],['training'=>$training])[0];
        $query = $query->reshape($full_output_shape);

        //echo "query_: ".$mo->toString($query,format:'%14.7f',indent:true)."\n";
    
        # `key` = [B, S, N, H]
        [$full_input_shape,$full_output_shape,$Fk] = $this->make_forward_dense_shape(
            $key, [$this->numHeads, $this->keyDim],
        );
        $key = $key->reshape($full_input_shape);
        $key = $this->key_dense->_rawCall([$key],['training'=>$training])[0];
        $key = $key->reshape($full_output_shape);

        # `value` = [B, S, N, H]
        [$full_input_shape,$full_output_shape,$Fv] = $this->make_forward_dense_shape(
            $value, [$this->numHeads, $this->valueDim],
        );
        $value = $value->reshape($full_input_shape);
        $value = $this->value_dense->_rawCall([$value],['training'=>$training])[0];
        $value = $value->reshape($full_output_shape);

        //echo "query after dense: ".$mo->toString($query,indent:true)."\n";
        //echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,indent:true)."\n";

        [$attention_output, $attention_scores, $scaled_query] = $this->compute_attention(
                $query, $key, $value,
                $query_mask,
                $value_mask,
                $key_mask,
                $attention_mask,
                $useCausalMask,
                $training,
        );
        $container->attention_output = $attention_output;

        # `attention_output` = [B, (Tq), H, valueDim]
        //echo "attention_output before dense: ".$mo->toString($attention_output,indent:true)."\n";
        //echo "attention_output before dense: ".$mo->shapeToString($attention_output->shape())."\n";
        [$full_input_shape,$full_output_shape,$dmy] = $this->make_forward_dense_shape(
            $attention_output, $Fq,
        );
        //echo "Fq: ".$mo->shapeToString($Fq)."\n";
        //echo "full_input_shape: ".$mo->shapeToString($full_input_shape)."\n";
        //echo "full_output_shape: ".$mo->shapeToString($full_output_shape)."\n";
        $attention_output = $attention_output->reshape($full_input_shape);
        $attention_output = $this->output_dense->_rawCall([$attention_output],['training'=>$training])[0];
        $attention_output = $attention_output->reshape($full_output_shape);
        //echo "attention_output after dense: ".$mo->toString($attention_output,indent:true)."\n";
        //echo "outputs          ";
        //if(!$mo->la()->isclose(
        //    $K->sum($attention_output,axis:-1),
        //    $mo->array([
        //        [132.64127, 152.00198, 167.6276,  179.13527, 187.18292, 192.69943],
        //        [710.6805 , 713.35693, 715.21  ,  716.51843, 717.4543 , 718.13116],                
        //    ]),
        //    debug:true,
        //)){
        //    throw new \Exception("Error Processing Request", 1);
        //}

        $container->attention_mask = $attention_mask;
        $container->training = $training;
        $container->scaled_query = $scaled_query;
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
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();
        //echo "dOutputs".$mo->shapeToString($dOutputs->shape()).": ";
        //echo $mo->toString($dOutputs,indent:true)."\n";
        $container = $this->container();

        // (B.Tq.Fq) => (B.Tq.H.Dv)
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dOutputs,[$this->numHeads, $this->valueDim],
        );
        //echo "dOutputs".$mo->shapeToString($dOutputs->shape())."\n";
        //echo "full_output_shape".$mo->shapeToString($full_output_shape)."\n";
        //echo "full_input_shape".$mo->shapeToString($full_input_shape)."\n";
        $dOutputs = $dOutputs->reshape($full_output_shape);
        $dAttention_output = $this->output_dense->_rawDifferentiate([$dOutputs])[0];
        $dAttention_output = $dAttention_output->reshape($full_input_shape);

        //echo "dAttention_output: ".$mo->toString($dAttention_output,indent:true)."\n";

        [$dQuery, $dKey, $dValue] = $this->compute_differntiate_attention(
            $dAttention_output,
            $container->scaled_query,
            $container->key,
            $container->value,
            $container->attention_output,
            $container->attention_scores,
            $container->attention_mask,
            $container->training
        );

        //echo "============================================\n";
        //echo "dQuery before dense->df(): ".$mo->toString($dQuery,indent:true)."\n";
        //echo "dValue_: ".$mo->toString($dValue,format:'%13.7e',indent:true)."\n";

        // (B.Tv.H.Dv) => (B.Tv.Fv)
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dValue, $this->value_feature_shape,
        );
        $dValue = $dValue->reshape($full_output_shape);
        $dValue = $this->value_dense->_rawDifferentiate([$dValue])[0];
        $dValue = $dValue->reshape($full_input_shape);

        // (B.Tq.H.Dk) => (B.Tq.Fq)
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dQuery, $this->query_feature_shape,
        );
        $dQuery = $dQuery->reshape($full_output_shape);
        $dQuery = $this->query_dense->_rawDifferentiate([$dQuery])[0];
        $dQuery = $dQuery->reshape($full_input_shape);

        // (B.Tk.H.Dk) => (B.Tk.Fk)
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dKey, $this->key_feature_shape,
        );
        $dKey = $dKey->reshape($full_output_shape);
        $dKey   = $this->key_dense->_rawDifferentiate([$dKey])[0];
        $dKey = $dKey->reshape($full_input_shape);

        //echo "============================================\n";
        //echo "dQuery after dense->df(): ".$mo->toString($dQuery,indent:true)."\n";
        //echo "dValue: ".$mo->toString($dValue,format:'%13.7e',indent:true)."\n";

        if($container->sameKey) {
            $K->update_add($dValue,$dKey);
            //echo "dValue: ".$mo->toString($dValue,format:'%13.7e',indent:true)."\n";
            return [$dQuery,$dValue];
        } else {
            return [$dQuery,$dValue,$dKey];
        }
    }

    protected function alloc_attention_mask(
        NDArray $query,
        NDArray $value,
    ) : NDArray
    {
        $K = $this->backend;
        $n_attn_axes = count($this->attention_axes);
        //$q_seq_length = $query->shape()[1];
        //$v_seq_length = ($value===null) ? $q_seq_length : $value->shape()[1];
        $q_seq_shape = array_slice($query->shape(),1,$n_attn_axes);
        $v_seq_shape = ($value===null) ? $q_seq_shape : array_slice($value->shape(),1,$n_attn_axes);
        $batches = $query->shape()[0];
        $shape = array_merge([$batches],$q_seq_shape,$v_seq_shape);
        $attention_mask = $K->ones($shape,dtype:NDArray::bool);
        return $attention_mask;
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
    private function compute_masked_attention_scores(
        NDArray $attention_scores,
        NDArray $query,
        NDArray $value,
        NDArray $query_mask=null,
        NDArray $value_mask=null,
        NDArray $key_mask=null,
        NDArray $attention_mask=null,
        bool $useCausalMask=null,
    ) : ?NDArray
    {
        //echo "==== compute_masked_attention_scores ====\n";
        $mo = $this->backend->localMatrixOperator();
        $K = $this->backend;

        if($value_mask && $key_mask) {
            if(spl_object_id($value_mask)==spl_object_id($key_mask)) {
                $key_mask = null;
            }
        }

        $auto_mask = null;
        if($attention_mask) {
            # merge attention_mask & automatic mask, to shape [B, T, S]
            //$attention_mask = [
            //    ($attention_mask===null) ? $auto_mask : ($K->cast($attention_mask, NDArray::bool) & $auto_mask)
            //];
            // [B,T,S] -expand-> [B, 1, T, S] -mask-to-> [B, H, T, S]
        }
        if($useCausalMask) {
            //echo "=== useCausalMask ====\n";
            #original <the shape of the causal mask is [1, T, S]>
            #rindow<the shape of the causal mask is [T, S]>
            // [T,S]
            $causal_mask = $this->compute_causal_mask($query, $value);
            if($auto_mask==null && $query_mask==null && $value_mask==null && $key_mask==null) {
                //echo "causal-mask".$K->localMatrixOperator()->shapeToString($mask->shape()).": ".$K->localMatrixOperator()->toString($mask,indent:true)."\n";
                //$auto_mask = ($auto_mask===null) ? $mask : ($auto_mask & $mask);
                //echo 'causal_mask='.$mo->shapeToString($causal_mask->shape())."\n";
                // [T,S] -expand-> [1, 1, T, S] -mask-to-> [B, H, T, S]
                $attention_scores = $K->masking($causal_mask,$attention_scores,fill:-1e9,mode:1,axis:-$causal_mask->ndim());
                //echo "causal_mask only\n";
                return $attention_scores;
            }
            if($auto_mask==null) {
                // [B,T,S]
                $auto_mask = $this->alloc_attention_mask($query,$value);
            }
            // [B, T, S] <==masking== [T,S]
            $K->update_masking($auto_mask,$causal_mask,axis:-$causal_mask->ndim());
        }
        if($query_mask) {
            //echo "=== query_mask ===\n";
            if($auto_mask==null && $value_mask==null && $key_mask==null) {
                #$query_mask = $K->cast($query_mask, NDArray::bool);
                ## B = batch size, T = max query length
                #$auto_mask = $K->expand_dims($query_mask, -1);  # shape is [B, T, 1]
                // [B,T] -expand-> [B, 1, T, 1] -mask-to-> [B, H, T, S]
                $attention_scores = $K->masking($query_mask,$attention_scores,fill:-1e9,mode:1,batchDims:1,axis:2);
                //echo "query_mask only\n";
                return $attention_scores;
            }
            if($auto_mask==null) {
                // [B,T,S]
                $auto_mask = $this->alloc_attention_mask($query,$value);
            }
            // [B,T,S] <==masking== [B,T]
            $K->update_masking($auto_mask,$query_mask);
        }
        if($value_mask) {
            //echo "=== value_mask ===\n";
            if($auto_mask==null && $key_mask==null) {
                #$value_mask = $K->cast($value_mask, NDArray::bool);
                ## B = batch size, S == max value length
                #$mask = $K->expand_dims($value_mask, -2);  # shape is [B, 1, S]
                #$auto_mask = ($auto_mask===null) ? $mask : ($auto_mask & $mask);
                // [B,S] -expand-> [B, 1, 1, S] -mask-to-> [B, H, T, S]
                $attention_scores = $K->masking($value_mask,$attention_scores,fill:-1e9,mode:1,batchDims:1,axis:-$value_mask->ndim()+1);
                //echo "value_mask only\n";
                return $attention_scores;
            }
            if($auto_mask==null) {
                // [B,T,S]
                $auto_mask = $this->alloc_attention_mask($query,$value);
            }
            // [B,T,S] <==masking== [B,S]
            $K->update_masking($auto_mask,$value_mask,batchDims:1,axis:-$value_mask->ndim()+1);
        }
        if($key_mask) {
            if($auto_mask==null) {
                //echo "=== key_mask ===\n";
                #$key_mask = $K->cast($key_mask, NDArray::bool);
                ## B == batch size, S == max key length == max value length
                #$mask = $K->expand_dims($key_mask, -2);  # shape is [B, 1, S]
                #$auto_mask = ($auto_mask===null) ? $mask : ($auto_mask & $mask);
                // [B,S] -expand-> [B, 1, 1, S] -mask-to-> [B, H, T, S]
                $attention_scores = $K->masking($key_mask,$attention_scores,fill:-1e9,mode:1,batchDims:1,axis:-$key_mask->ndim()+1);
                //echo "key_mask only\n";
                return $attention_scores;
            }
            if($auto_mask==null) {
                // [B,T,S]
                $auto_mask = $this->alloc_attention_mask($query,$value);
            }
            // [B,T,S] <==masking== [B,S]
            $K->update_masking($auto_mask,$key_mask,batchDims:1,axis:-$key_mask->ndim()+1);
        }
        if($auto_mask) {
            // [B,H,T,S] <==masking== [B,T,S]
            $attention_scores = $K->masking($auto_mask,$attention_scores,fill:-1e9,mode:1,batchDims:1,axis:-$auto_mask->ndim()+1);
            //echo "auto_mask\n";
        }
        ## attention_mask = [B, T, S]
        #echo "compute_attention_mask: (".(isset($attention_mask)?
        #    implode(',',$attention_mask->shape()):'null').")\n";
        #return [
        #    $attention_mask,
        #    [$query_mask,$value_mask,$key_mask],
        #    $causal_mask,
        #];
        return $attention_scores;
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
        //echo "compute_causal_mask(\n";
        //echo "  query: (".implode(',',$query->shape()).")\n";
        //echo "  value: (".implode(',',$value->shape()).")\n";

        $K = $this->backend;
        $n_attn_axes = count($this->attention_axes);
        //$q_seq_length = $query->shape()[1];
        //$v_seq_length = ($value===null) ? $q_seq_length : $value->shape()[1];
        $q_seq_shape = array_slice($query->shape(),1,$n_attn_axes);
        $v_seq_shape = ($value===null) ? $q_seq_shape : array_slice($value->shape(),1,$n_attn_axes);
        $q_seq_length = array_product($q_seq_shape);
        $v_seq_length = array_product($v_seq_shape);
        $ones_mask = $K->ones([$q_seq_length, $v_seq_length],dtype:NDArray::bool);
        $causal_mask = $K->bandpart($ones_mask,-1,0);
        $causal_mask = $causal_mask->reshape(array_merge($q_seq_shape,$v_seq_shape));
        //echo "  causal_mask: (".implode(',',$causal_mask->shape()).")\n";
        //echo ")\n";
        return $causal_mask;
        //$row_index = $K->cumsum($ones_mask, axis:-2);
        //echo "======row_index======\n";
        //echo $K->localMatrixOperator()->toString($row_index,indent:true)."\n";
        //$col_index = $K->cumsum($ones_mask, axis:-1);
        //echo "======col_index======\n";
        //echo $K->localMatrixOperator()->toString($col_index,indent:true)."\n";
        //$mask = $K->sub($row_index, $col_index);
        //echo "======sub======\n";
        //echo $K->localMatrixOperator()->toString($mask,indent:true)."\n";
        //return $K->greaterEqual($mask, 0);
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
        int $rank,          // full rank
        array $attn_axes    // full attn_axes
    ) : array
    {
        $target_notation = "";
        //$full_rank = $rank;
        for($i=0; $i<$rank; ++$i) {
            $target_notation .= $this->index_to_einsum_variable($i);
        }
        //echo "target_notation=$target_notation\n";
        # `batch_dims` includes the head dim.
        $batch_dims = range(0,$rank-1);
        //echo "batch_dims=(".implode(',',$batch_dims).")\n";
        $delete = array_merge($attn_axes, [$rank-1]);
        //echo "delete=[".implode(',',$delete)."]\n";
        foreach($delete as $i) {
            unset($batch_dims[$i]);
        }
        //echo "deleted batch_dims=(".implode(',',$batch_dims).")\n";
        //echo "attn_axes=(".implode(',',$attn_axes).")\n";
        $letter_offset = $rank;
        $source_notation = "";
        for($i=0;$i<$rank;++$i) {
            if(in_array($i,$batch_dims) || $i == $rank-1) {
                $source_notation .= $target_notation[$i];
            } else {
                $source_notation .= $this->index_to_einsum_variable($letter_offset);
                $letter_offset++;
            }
        }
    
        $product_notation = implode('',array_merge(
            array_map(fn($i)=>$target_notation[$i],$batch_dims),
            array_map(fn($i)=>$target_notation[$i],$attn_axes),
            array_map(fn($i)=>$source_notation[$i],$attn_axes)
        ));
        //echo "source_notation=$source_notation\n";
        //echo "target_notation=$target_notation\n";
        //echo "product_notation=$product_notation\n";
        [
            $dot_product_equation,
            $backward_dot_product_key_equation,
            $backward_dot_product_query_equation,
        ] = $this->generate_equations(
            $source_notation,  // key
            $target_notation,  // query
            $product_notation, // scores
        );
        $attn_scores_rank = strlen($product_notation);
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
        //echo "equation:$equation, bias_axes:$bias_axes, output_rank:".(strlen($output_str))."]\n";
        return [$equation, $bias_axes, strlen($output_str)];
    }

    protected function build_dense_args(
        array $inputShape,
        array $effectorDims,
    ) : array
    {
        if($this->attention_axes==null) {
            $this->attention_axes = range(1, count($inputShape)-2);
        }

        // input:   ((Batch, (T)), (Feature))
        // kernel:  ((Feature), （Heads)
        // output:  ((Batch, (T)), (Heads))

        // inputShape = (batch,(T),dim)
        // units = numHeads*headDim
        // dense_input_shape = (T,dim)
        $feature = $inputShape;
        $batch = array_shift($feature);
        $T = array_splice($feature,0,count($this->attention_axes));
        $units = array_product($effectorDims);
        $dense_input_shape = [array_product($T),array_product($feature)];
        return [$batch,$T,$feature,$units,$dense_input_shape];
    }

    protected function make_forward_dense_shape(
        NDArray $inputs,
        array $effectorDims,
    ) : array
    {
        $feature = $inputs->shape();
        $batch = array_shift($feature);
        $T = array_splice($feature,0,count($this->attention_axes));
        $dense_input_shape = [$batch*array_product($T),array_product($feature)];
        $full_output_shape = array_merge([$batch],$T,$effectorDims);
        return [$dense_input_shape,$full_output_shape,$feature];
    }

    protected function make_backward_dense_shape(
        NDArray $dOutputs,
        array $feature,
    ) : array
    {
        $effectorDims = $dOutputs->shape();
        $batch = array_shift($effectorDims);
        $T = array_splice($effectorDims,0,count($this->attention_axes));
        //echo "T:".implode(',',$T)."\n";
        //echo "effectorDims:".implode(',',$effectorDims)."\n";
        $full_output_shape = [$batch*array_product($T),array_product($effectorDims)];
        $full_input_shape = array_merge([$batch],$T,$feature);
        return [$full_output_shape,$full_input_shape];
    }

    
    private function get_output_shape(
        int $output_rank,
        array $input_shape,
        array $known_last_dims
        ) : array
    {
        //echo "get_output_shape(\n".
        //    "  output_rank=$output_rank,\n".
        //    "  input_shape=(".implode(',',$input_shape)."),\n".
        //    "  known_last_dims=(".implode(',',$known_last_dims).")\n".
        //    ")\n";
        $output_shape = array_merge(
            array_slice($input_shape,1,($output_rank - count($known_last_dims))),
            $known_last_dims
        );

        // ** CAUTION **
        // output_shape is without batch dims
        // this is the spec from the original Tensorflow EinsumDense
        //echo "get_output_shape=(".implode(',',$output_shape).")\n";
        return $output_shape;
    }
}
