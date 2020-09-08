<?php
namespace Rindow\NeuralNetworks\Backend\RindowBlas;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Backend
{
    protected $initializers = [
        'glorot_uniform'    => 'glorot_uniform',
        'glorot_normal'     => 'glorot_normal',
        'he_uniform'        => 'he_uniform',
        'he_normal'         => 'he_normal',
        'random_uniform'    => 'random_uniform',
        'random_normal'     => 'random_normal',
        'orthogonal'        => 'orthogonal',
        'zeros'             => 'zeros',
        'ones'              => 'ones',
    ];
    protected $epsilon = 1e-7;
    protected $equalEpsilon = 1e-06;
    protected $matrixOperator;
    protected $la;

    public function __construct($matrixOperator)
    {
        $this->matrixOperator = $matrixOperator;
        $this->la = $matrixOperator->laRawMode();
    }

    public function epsilon()
    {
        return $this->epsilon;
    }

    public function setEpsilon($epsilon) : void
    {
        $this->epsilon = $epsilon;
    }

    public function dtypeToString($dtype) : string
    {
        $mo = $this->matrixOperator;
        return $mo->dtypeToString($dtype);
    }

    public function alloc(array $shape,$dtype=null)
    {
        return $this->la->alloc($shape,$dtype);
    }

    public function allocLike(NDArray $x)
    {
        return $this->la->alloc($x->shape(),$x->dtype());
    }

    public function array($value, $dtype=null)
    {
        return $this->matrixOperator->array(
            $value,$dtype);
    }

    public function getInitializer($name)
    {
        if(!array_key_exists($name,$this->initializers))
            throw new InvalidArgumentException('Unsupported initializer: '.$name);
        return [$this,$this->initializers[$name]];
    }

    public function glorot_normal(array $shape,$nodeNum=null)
    {
        $mo = $this->matrixOperator;
        if($nodeNum===null){
            $tmpShape = $shape;
            $nodeNum = [array_shift($tmpShape)];
            $nodeNum[] = array_product($tmpShape);
        }
        [$fanIn,$fanOut]=$nodeNum;
        $scale = 1/max(($fanIn+$fanOut)/2.0, 1.0);
        # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        $scale = sqrt($scale) / 0.87962566103423978;
        $kernel = $this->la->randomNormal($shape,0.0,$scale);
        return $kernel;
    }

    public function glorot_uniform(array $shape,$nodeNum=null)
    {
        $mo = $this->matrixOperator;
        if($nodeNum===null){
            $tmpShape = $shape;
            $nodeNum = [array_shift($tmpShape)];
            $nodeNum[] = array_product($tmpShape);
        }
        [$fanIn,$fanOut]=$nodeNum;
        $scale = 1/max(($fanIn+$fanOut)/2.0, 1.0);
        $limit = sqrt(3*$scale);
        $kernel = $this->la->randomUniform($shape,-$limit,$limit);
        return $kernel;
    }

    public function random_normal(array $shape,$nodeNum=null)
    {
        $mo = $this->matrixOperator;
        $mean=0.0;
        $stddev=0.05;
        if(is_array($nodeNum)) {
            if(array_key_exists('mean',$nodeNum)){
                $mean = $nodeNum['mean'];
            }
            if(array_key_exists('stddev',$nodeNum)){
                $stddev = $nodeNum['stddev'];
            }
        }
        $kernel = $this->la->randomNormal($shape,$mean,$stddev);
        return $kernel;
    }

    public function random_uniform(array $shape,$nodeNum=null)
    {
        $mo = $this->matrixOperator;
        $minval=-0.05;
        $maxval=0.05;
        if(is_array($nodeNum)) {
            if(array_key_exists('minval',$nodeNum)){
                $minval = $nodeNum['minval'];
            }
            if(array_key_exists('maxval',$nodeNum)){
                $maxval = $nodeNum['maxval'];
            }
        }
        $kernel = $this->la->randomUniform($shape,$minval,$maxval);
        return $kernel;
    }

    public function orthogonal(array $shape,$nodeNum=null)
    {
        $tmpShape = $shape;
        $num_cols = array_pop($tmpShape);
        $num_rows = (int)array_product($tmpShape);
        $flat_shape = [$num_rows,$num_cols];
        $a = $this->la()->randomNormal(
            $flat_shape);
        [$u,$s,$vt] = $this->la()->svd($a);
        # Pick the one with the correct shape.
        $q = ($u->shape()==$flat_shape)?$u: $v;
        $q = $q->reshape($shape);
        return $this->la()->slice($q,[0,0], [$shape[0],$shape[1]]);
    }


    public function he_normal(array $shape,$nodeNum=null)
    {
        $mo = $this->matrixOperator;
        if($nodeNum===null){
            $tmpShape = $shape;
            $nodeNum = [array_shift($tmpShape),0.05];
        }
        [$fanIn,$fanOut]=$nodeNum;
        $scale = 2/max($fanIn, 1.0);
        # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        $scale = sqrt($scale) / 0.87962566103423978;
        $kernel = $this->la->randomNormal($shape,0.0,$scale);
        return $kernel;
    }

    public function he_uniform(array $shape,$nodeNum=null)
    {
        $mo = $this->matrixOperator;
        if($nodeNum===null){
            $tmpShape = $shape;
            $nodeNum = [array_shift($tmpShape),0.05];
        }
        [$fanIn,$fanOut]=$nodeNum;
        $scale = 2/max($fanIn, 1.0);
        $limit = sqrt(3*$scale);
        $kernel = $this->la->randomUniform($shape,-$limit,$limit);
        return $kernel;
    }

    public function zeros(array $shape,$dtype=null)
    {
        $x = $this->la->alloc($shape,$dtype);
        $this->la->zeros($x);
        return $x;
    }

    public function ones(array $shape,$dtype=null)
    {
        $x = $this->la->alloc($shape,$dtype);
        $this->la->zeros($x);
        $this->la->increment($x,1.0);
        return $x;
    }

    public function zerosLike(NDArray $x)
    {
        $y = $this->la->alloc($x->shape(),$x->dtype());
        $this->la->zeros($y);
        return $y;
    }

    public function onesLike(NDArray $x)
    {
        $y = $this->la->alloc($x->shape());
        $this->la->zeros($y);
        $this->la->increment($y,1.0);
        return $y;
    }

    public function clear(NDArray $x)
    {
        $this->la->zeros($x);
        return $x;
    }

    public function copy(NDArray $from,NDArray $to=null)
    {
        return $this->la->copy($from, $to);
    }

    public function cast(NDArray $x,$dtype)
    {
        $mo = $this->matrixOperator;
        return $mo->astype($x,$dtype);
    }

    public function transpose(NDArray $x)
    {
        $mo = $this->matrixOperator;
        return $mo->transpose($x);
    }

    public function add(NDArray $x, NDArray $y)
    {
        $la = $this->la;
        $ndimX = $x->ndim();
        $ndimY = $y->ndim();
        if($ndimX == $ndimY) {
            $x = $la->copy($x,$la->alloc($y->shape(),$x->dtype()));
            $la->axpy($y,$x);
            return $x;
        } elseif($ndimX < $ndimY) {
            $x = $la->duplicate($x,null,null,$la->alloc($y->shape(),$x->dtype()));
            $la->axpy($y,$x);
            return $x;
        } else {
            $y = $la->duplicate($y,null,null,$la->alloc($x->shape(),$y->dtype()));
            $la->axpy($x,$y);
            return $y;
        }
    }

    public function sub(NDArray $x, NDArray $y)
    {
        $la = $this->la;
        $ndimX = $x->ndim();
        $ndimY = $y->ndim();
        if($ndimX == $ndimY) {
            $x = $la->copy($x,$la->alloc($y->shape(),$x->dtype()));
            $la->axpy($y,$x,-1.0);
            return $x;
        } elseif($ndimX < $ndimY) {
            $x = $la->duplicate($x,null,null,$la->alloc($y->shape(),$x->dtype()));
            $la->axpy($y,$x,-1.0);
            return $x;
        } else {
            $y = $la->duplicate($y,null,null,$la->alloc($x->shape(),$y->dtype()));
            $la->increment($y,0,-1);
            $la->axpy($x,$y);
            return $y;
        }
    }

    public function mul(NDArray $x, NDArray $y)
    {
        $la = $this->la;
        if($x->ndim() < $y->ndim()) {
            $y = $la->copy($y);
            return $la->multiply($x,$y);
        } else {
            $x = $la->copy($x);
            return $la->multiply($y,$x);
        }
    }

    public function div(NDArray $x, NDArray $y)
    {
        $la = $this->la;
        $y = $la->copy($y);
        $la->reciprocal($y);
        if($x->ndim() < $y->ndim()) {
            return $la->multiply($x,$y);
        } else {
            $m = $la->copy($x);
            return $la->multiply($y,$x);
        }
    }

    public function update(NDArray $x, NDArray $newX) : NDArray
    {
        $this->la->copy($newX,$x);
        return $x;
    }

    public function update_add(NDArray $x, NDArray $increment) : NDArray
    {
        $this->la->axpy($increment,$x);
        return $x;
    }

    public function update_sub(NDArray $x, NDArray $decrement) : NDArray
    {
        $this->la->axpy($decrement,$x,-1.0);
        return $x;
    }

    public function update_mul(NDArray $x, NDArray $magnifications) : NDArray
    {
        return $this->la->multiply($magnifications,$x);
    }

    public function scale(float $a, NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->scal($a, $x);
    }

    public function increment(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->increment($x, $a);
    }

    public function pow(NDArray $x, float $y)
    {
        $x = $this->la->copy($x);
        return $this->la->pow($x,$y);
    }

    public function square(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->square($x);
    }

    public function sqrt(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->sqrt($x);
    }

    public function rsqrt(NDArray $x,float $beta=null, float $alpha=null)
    {
        $x = $this->la->copy($x);
        return $this->la->rsqrt($x,$beta,$alpha);
    }

    public function abs(NDArray $x)
    {
        return $this->matrixOperator->f('abs',$x);
    }

    public function maximum(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->maximum($a,$x);
    }

    public function minimum(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->minimum($a,$x);
    }

    public function greater(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->greater($a,$x);
    }

    public function less(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->less($a,$x);
    }

    public function exp(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->exp($x);
    }

    public function log(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->log($x);
    }

    public function equal($x,$y)
    {
        return $this->la->equal($x,$y);
    }

    public function tanh($x)
    {
        return $this->la->tanh($x);
    }

    public function sum(NDArray $x, int $axis=null)
    {
        if($axis===null) {
            return $this->la->sum($x);
        } else {
            return $this->la->reduceSum($x,$axis);
        }
    }

    public function mean(NDArray $x,int $axis=null)
    {
        if($axis===null) {
            return $this->la->sum($x) / $x->size();
        } else {
            return $this->la->reduceMean($x,$axis);
        }
    }

    public function max(NDArray $x,int $axis=null)
    {
        if($axis===null) {
            return $this->la->max($x);
        } else {
            return $this->la->reduceMax($x,$axis);
        }
    }

    public function min(NDArray $x,int $axis=null)
    {
        $mo = $this->matrixOperator;
        return $mo->min($x,$axis);
    }

    public function argMax(NDArray $x,int $axis=null,$dtype=null)
    {
        if($axis===null) {
            return $this->la->imax($x);
        } else {
            return $this->la->reduceArgMax($x,$axis,null,$dtype);
        }
    }

    public function argMin(NDArray $x,int $axis=null)
    {
        $mo = $this->matrixOperator;
        return $mo->argMin($x,$axis);
    }

    public function rand($shape)
    {
        $mo = $this->matrixOperator;
        return $this->randomUniformVariables($shape,0.0,1.0);
    }

    public function randomChoice($a,int $size=null, bool $replace=null)
    {
        $mo = $this->matrixOperator;
        return $mo->random()->choice($a, $size, $replace);
    }

    public function randomSequence(int $base, int $size=null, int  $seed=null)
    {
        $mo = $this->matrixOperator;
        return $this->la->randomSequence($base, $size, $seed);
    }

    public function dot($x,$y)
    {
        return $this->la->gemm($x,$y);
    }

    public function gemm(NDArray $a,NDArray $b,float $alpha=null,float $beta=null,NDArray $c=null,$transA=null,$transB=null)
    {
        return $this->la->gemm($a, $b,$alpha,$beta,$c,$transA,$transB);
    }

    public function batch_gemm(NDArray $a,NDArray $b,float $alpha=null,float $beta=null,NDArray $c=null)
    {
        $batchSize = $a->shape()[0];
        if($c==null) {
            $outputs = null;
        } else {
            $outputs = $this->alloc(array_merge([$batchSize],$c->shape()));
            $this->la->duplicate($c,null,null,$outputs);
        }
        return $this->la->gemm($a, $b, $alpha, $beta, $outputs);
    }

    public function select(NDArray $source,NDArray $selector,$axis=null)
    {
        return $this->la->select($source,$selector,$axis);
    }

    public function scatter(NDArray $indices,NDArray $values,int  $numClass,$axis=null,NDArray $target=null)
    {
        return $this->la->scatter($indices,$values,$numClass,$axis,$target);
    }

    public function scatterAdd(NDArray $target,NDArray $indices,NDArray $values,$axis=null)
    {
        return $this->la->scatterAdd($indices,$values,$target,$axis);
    }

    public function slice(
        NDArray $input,
        array $begin, array $size,
        NDArray $output=null) {
        return $this->la->slice(
            $input,
            $begin,$size,
            $output);
    }

    public function stick(
        NDArray $input,
        NDArray $output,
        array $begin, array $size
        ) {
        return $this->la->stick(
            $input,
            $output,
            $begin,$size
            );
    }

    public function stack(
        array $inputs,
        int $axis=null
        ) {
        return $this->la->stack(
            $inputs,
            $axis
            );
    }

    public function repeat(
        NDArray $inputs,
        int $repeats
        ) {
        if($inputs->ndim()!=2) {
            throw new InvalidArgumentException('inputs dimension must be 2D');
        }
        return $this->la->repeat(
            $inputs,
            $repeats
            );
    }

    public function reduceSumRepeated(
        NDArray $inputs
        ) {
        if($inputs->ndim()!=3) {
            throw new InvalidArgumentException('inputs dimension must be 3D');
        }
        return $this->la->reduceSumRepeated(
            $inputs
            );
    }

    public function oneHot(NDArray $indices, int $numClass) : NDArray
    {
        if($indices->ndim()!=1) {
            throw new InvalidArgumentException('indices must be 1-D NDarray');
        }
        return $this->la->onehot($indices,$numClass);
    }

    public function randomUniformVariables(array $shape, $low, $high, $dtype=null, int $seed=null, NDArray $x=null) : NDArray
    {
        return $this->la->randomUniform($shape,$low,$high,$dtype,$seed,$x);
    }

    public function randomNormalVariables(array $shape, $mean, $scale, $dtype=null, int $seed=null, NDArray $x=null) : NDArray
    {
        return $this->la->randomNormal($shape,$mean,$scale,$dtype,$seed,$x);
    }

    public function relu($x) : NDArray
    {
        return $this->maximum($x,0.0);
    }

    public function sigmoid(NDArray $inputs) : NDArray
    {
        $la = $this->la;
        //  1 / (1.0+exp(-$x))
        $X = $la->copy($inputs);
        return $la->reciprocal(
                $la->exp($la->scal(-1.0,$X)),1.0);
    }

    public function dSigmoid(NDArray $dOutputs, NDArray $outputs) : NDArray
    {
        $la = $this->la;
        // dx = dy * ( 1 - y ) * y
        return $la->multiply($dOutputs,$la->multiply($outputs,
            $la->increment($la->copy($outputs),1.0,-1.0)));
    }

    public function softmax(NDArray $X) : NDArray
    {
        $la = $this->la;
        // Yk = exp(Ak + C') / sum(exp(Ai + C'))
        $ndim = $X->ndim();

        if($ndim == 1) {
            //$X = $this->la->increment($this->la->copy($X), -$this->la->max($X)); # fix overflow
            //$expX = $this->la->exp($X);
            //return $this->la->scal(1/$this->la->sum($expX),$expX);

            // Native softmax function !!!
            return $la->softmax($la->copy($X)->reshape([1,$X->size()]))
                ->reshape([$X->size()]);
        } else {
            $orig = $shape = $X->shape();
            $inputDim = array_pop($shape);
            $X = $X->reshape([(int)array_product($shape),$inputDim]);
            $y = $la->softmax($la->copy($X));
            return $y->reshape($orig);
        }
        /*
        if($ndim == 2) {

            //$X = $this->la->add($this->la->reduceMax($X, $axis=1),
            //                    $this->la->copy($X),-1,$trans=true);  # fix overflow
            //$expX = $this->la->exp($X);
            //$Y = $this->la->multiply(
            //    $this->la->reciprocal($this->la->reduceSum($expX, $axis=1)),
            //    $expX,$trans=true);
            //return $Y;

            // Native softmax function !!!
            return $la->softmax($la->copy($X));

        } else {
            throw new InvalidArgumentException('Array must be 1-D or 2-D.');
        }
        */
    }

    public function dSoftmax(NDArray $dOutputs, NDArray $outputs) : NDArray
    {
        $la = $this->la;
        if($dOutputs->shape()!=$outputs->shape()){
            throw new InvalidArgumentException('unmatch predicts shape: ['.implode(',',$predicts->shape()).']');
        }

        // softmax:  yk      = exp(ak) / sumj(exp(aj))
        // dsoftmax: dyk/daj =  yk * (1 - yj): j=k , -yk * yj : j!=k
        //                   =  yk * (I(kj) - yj)  ; I(kj) -> 1:k=j, 0:k!=j
        //                   =

        // dx = (y * dy) - sum(y * dy) * y
        $dx = $la->multiply($outputs, $la->copy($dOutputs));
        $dInputs = $la->axpy(
            $la->multiply($la->reduceSum($dx, $axis=1),
                                $la->copy($outputs),$trans=true),$dx,-1.0);
        //$dInputs = $this->la->scal(1/$dOutputs->shape()[0],$dInputs);
        return $dInputs;
    }

    public function conv1d(
        object $status,
        NDArray $inputs,
        NDArray $kernel,
        NDArray $bias=null,
        array $strides=null,
        string $padding=null,
        string $data_format=null
        ) : NDArray
    {
        $rank = 1;
        return $this->doConv(
            $rank,
            $status,
            $inputs,
            $kernel,
            $bias,
            $strides,
            $padding,
            $data_format
        );
    }

    public function dConv1d(
        object $status,
        NDArray $dOutputs,
        NDArray $dKernel,
        NDArray $dBias=null
        ): NDArray
    {
        $rank = 1;
        return $this->doDConv(
            $rank,
            $status,
            $dOutputs,
            $dKernel,
            $dBias
        );
    }

    public function pool1d(
        object $status,
        NDArray $inputs,
        array $poolSize,
        array $strides=null,
        string $padding=null,
        string $data_format=null,
        string $pool_mode=null
        ) : NDArray
    {
        $rank = 1;
        return $this->doPool(
            $rank,
            $status,
            $inputs,
            $poolSize,
            $strides,
            $padding,
            $data_format,
            $pool_mode
        );
    }

    public function dPool1d(
        object $status,
        NDArray $dOutputs
        ): NDArray
    {
        $rank = 1;
        return $this->doDPool(
            $rank,
            $status,
            $dOutputs
        );
    }

    public function conv2d(
        object $status,
        NDArray $inputs,
        NDArray $kernel,
        NDArray $bias=null,
        array $strides=null,
        string $padding=null,
        string $data_format=null
        ) : NDArray
    {
        $rank = 2;
        return $this->doConv(
            $rank,
            $status,
            $inputs,
            $kernel,
            $bias,
            $strides,
            $padding,
            $data_format
        );
    }

    public function dConv2d(
        object $status,
        NDArray $dOutputs,
        NDArray $dKernel,
        NDArray $dBias=null
        ): NDArray
    {
        $rank = 2;
        return $this->doDConv(
            $rank,
            $status,
            $dOutputs,
            $dKernel,
            $dBias
        );
    }

    public function pool2d(
        object $status,
        NDArray $inputs,
        array $poolSize,
        array $strides=null,
        string $padding=null,
        string $data_format=null,
        string $pool_mode=null
        ) : NDArray
    {
        $rank = 2;
        return $this->doPool(
            $rank,
            $status,
            $inputs,
            $poolSize,
            $strides,
            $padding,
            $data_format,
            $pool_mode
        );
    }

    public function dPool2d(
        object $status,
        NDArray $dOutputs
        ): NDArray
    {
        $rank = 2;
        return $this->doDPool(
            $rank,
            $status,
            $dOutputs
        );
    }

    public function conv3d(
        object $status,
        NDArray $inputs,
        NDArray $kernel,
        NDArray $bias=null,
        array $strides=null,
        string $padding=null,
        string $data_format=null
        ) : NDArray
    {
        $rank = 3;
        return $this->doConv(
            $rank,
            $status,
            $inputs,
            $kernel,
            $bias,
            $strides,
            $padding,
            $data_format
        );
    }

    public function dConv3d(
        object $status,
        NDArray $dOutputs,
        NDArray $dKernel,
        NDArray $dBias=null
        ): NDArray
    {
        $rank = 3;
        return $this->doDConv(
            $rank,
            $status,
            $dOutputs,
            $dKernel,
            $dBias
        );
    }

    public function pool3d(
        object $status,
        NDArray $inputs,
        array $poolSize,
        array $strides=null,
        string $padding=null,
        string $data_format=null,
        string $pool_mode=null
        ) : NDArray
    {
        $rank = 3;
        return $this->doPool(
            $rank,
            $status,
            $inputs,
            $poolSize,
            $strides,
            $padding,
            $data_format,
            $pool_mode
        );
    }

    public function dPool3d(
        object $status,
        NDArray $dOutputs
        ): NDArray
    {
        $rank = 3;
        return $this->doDPool(
            $rank,
            $status,
            $dOutputs
        );
    }

    protected function doConv(
        int $rank,
        object $status,
        NDArray $inputs,
        NDArray $kernel,
        NDArray $bias=null,
        array $strides=null,
        string $padding=null,
        string $data_format=null
        ) : NDArray
    {
        if($inputs->ndim()!=$rank+2) {
            throw new InvalidArgumentException('inputs must be '.($rank+2).'D NDArray');
        }
        $filterSize = $kernel->shape();
        $filters = array_pop($filterSize);
        $channels = array_pop($filterSize);
        $filterSize = array_values($filterSize);
        if($data_format == null ||
           $data_format=='channels_last') {
            $channels_first = false;
        } elseif($data_format=='channels_first') {
            $channels_first = true;
        } else {
            throw new InvalidArgumentException('$data_format must be channels_last or channels_first');
        }
        if($padding==null||
           $padding=='valid') {
            $padding=false;
        } elseif($padding=='same') {$
            $padding=true;
        } else {
            throw new InvalidArgumentException('padding must be valid or same');
        }
        $cols = $this->la->im2col(
            $inputs,
            $filterSize,
            $strides,
            $padding,
            $channels_first
        );
        $outShape = [];
        $shape = $cols->shape();
        $batches = array_shift($shape);
        for($i=0;$i<$rank;$i++){
            $outShape[] = array_shift($shape);
        }
        $cols =
            $cols->reshape([$batches*array_product($outShape),
            array_product($filterSize)*$channels]);
        $kernel = $kernel->reshape(
            [array_product($filterSize)*$channels,
             $filters]);

        if($bias){
            $outputs = $this->batch_gemm(
                $cols,
                $kernel,
                1.0,1.0,
                $bias
            );
        } else {
            $outputs = $this->gemm(
                $cols,
                $kernel
            );
        }

        $status->inputsShape = $inputs->shape();
        $status->kernel = $kernel;
        $status->cols = $cols;
        $status->flatten_out_shape =
            $outputs->shape();
        $status->filterSize = $filterSize;
        $status->strides = $strides;
        $status->padding = $padding;
        $status->channels_first = $channels_first;

        return $outputs->reshape(
            array_merge([$batches],$outShape,[$filters])
        );
    }

    protected function doDConv(
        int $rank,
        object $status,
        NDArray $dOutputs,
        NDArray $dKernel,
        NDArray $dBias=null
        ): NDArray
    {
        if($dOutputs->ndim()!=$rank+2) {
            throw new InvalidArgumentException('dOutputs must be '.($rank+2).'D NDArray');
        }
        $dCols = $this->zerosLike(
            $status->cols);
        $dOutputs = $dOutputs->reshape(
            $status->flatten_out_shape);
        $this->gemm(
            $dOutputs,
            $status->kernel,
            1.0,0.0,
            $dCols,
            false,true);

        // update params
        $this->gemm(
            $status->cols,
            $dOutputs,
            1.0,0.0,
            $dKernel->reshape($status->kernel->shape()),
            true,false);
        if($dBias){
            $this->copy($this->sum($dOutputs, $axis=0),$dBias);
        }

        $dInputs = $this->zeros($status->inputsShape);
        $this->la->col2im(
            $dCols,
            $dInputs,
            $status->filterSize,
            $status->strides,
            $status->padding,
            $status->channels_first
        );
        return $dInputs;
    }

    protected function doPool(
        int $rank,
        object $status,
        NDArray $inputs,
        array $poolSize,
        array $strides=null,
        string $padding=null,
        string $data_format=null,
        string $pool_mode=null
        ) : NDArray
    {
        if($inputs->ndim()!=$rank+2) {
            throw new InvalidArgumentException('inputs must be '.($rank+2).'D NDArray');
        }
        if($strides==null) {
            $strides=$poolSize;
        }
        $tmp = $inputs->shape();
        $batches = array_shift($tmp);
        if($data_format == null ||
           $data_format=='channels_last') {
            $channels_first = false;
            $channels = array_pop($tmp);
        } elseif($data_format=='channels_first') {
            $channels_first = true;
            $channels = array_shift($tmp);
        } else {
            throw new InvalidArgumentException('$data_format must be channels_last or channels_first');
        }
        if($padding==null||
           $padding=='valid') {
            $padding=false;
        } elseif($padding=='same') {$
            $padding=true;
        } else {
            throw new InvalidArgumentException('padding must be valid or same');
        }
        $cols = $this->la->im2col(
            $inputs,
            $poolSize,
            $strides,
            $padding,
            $channels_first,
            $cols_channels_first=true
        );
        $tmp = $cols->shape();
        $batches = array_shift($tmp);
        $outShape = [];
        for($i=0;$i<$rank;$i++){
            $outShape[] = array_shift($tmp);
        }
        $channels = array_shift($tmp);
        $filterSize = [];
        for($i=0;$i<$rank;$i++){
            $filterSize[] = array_shift($tmp);
        }
        $cols =
            $cols->reshape([$batches*array_product($outShape)*$channels,
        array_product($filterSize)    ]);

        if($pool_mode==null ||
            $pool_mode=='max') {
            $outputs = $this->la->reduceMax(
                $cols,$axis=1
            );
        } elseif($pool_mode=='avg') {
            $outputs = $this->la->reduceMean(
                $cols,$axis=1
            );
        } else {
            throw new InvalidArgumentException('pool_mode must be max or avg');
        }
        $status->inputsShape = $inputs->shape();
        $status->cols = $cols;
        $status->flatten_out_shape = $outputs->shape();
        $status->poolSize = $poolSize;
        $status->strides = $strides;
        $status->padding = $padding;
        $status->channels_first = $channels_first;
        $status->pool_mode = $pool_mode;
        $outputs = $outputs->reshape(
            array_merge([$batches],
            $outShape,
            [$channels]
            ));
        return $outputs;
    }

    protected function doDPool(
        int $rank,
        object $status,
        NDArray $dOutputs
        ): NDArray
    {
        if($dOutputs->ndim()!=$rank+2) {
            throw new InvalidArgumentException('dOutputs must be '.($rank+2).'D NDArray');
        }
        if($status->pool_mode=='avg'){
            // d mean
            // dx = repeat(dy/N)
            $num = $status->cols->shape()[1];
            $tmp = $this->la->scal(1/$num,$this->copy($dOutputs));
            $dCols = $this->la->duplicate($tmp,$num,$trans=true);
        } else {
            // d max
            //dx = dy * onehot(argMax(x))
            $argMax = $this->la->reduceArgMax(
                $status->cols,$axis=1);
            /*
            $dCols = $this->la->onehot(
                $argMax,
                array_product($status->poolSize));
            $dOutputs = $dOutputs->reshape(
                $status->flatten_out_shape
            );
            $this->la->multiply(
                $dOutputs,$dCols,$trans=true);
            */
            $dCols = $this->la->scatter(
                $argMax,
                $dOutputs->reshape([$dOutputs->size()]),
                array_product($status->poolSize),
                $axis=1
            );
        }

        $dInputs = $this->zeros(
            $status->inputsShape);
        $this->la->col2im(
            $dCols,
            $dInputs,
            $status->poolSize,
            $status->strides,
            $status->padding,
            $status->channels_first,
            $cols_channels_first=true
        );
        return $dInputs;
    }

    public function calcConvOutputShape(
        array $inputShape,
        array $filterSize,
        array $strides,
        string $padding=null,
        string $data_format=null
        ) : array
    {
        if($padding=='same') {
            return $inputShape;
        }
        if($data_format==null||
            $data_format=='channels_last') {
            $channels_first = false;
        } elseif($data_format=='channels_first') {
            $channels_first = true;
        } else {
            throw new InvalidArgumentException('data_format must be channels_last or channels_first');
        }
        if($channels_first) {
            $channels = array_unshift($inputShape);
        } else {
            $channels = array_pop($inputShape);
        }
        $inputShape = array_values($inputShape);
        foreach($inputShape as $idx=>$value) {
            $outputShape[$idx] = intval(floor(($inputShape[$idx]-$filterSize[$idx])/$strides[$idx])+1);
        }
        $outputShape = array_values($outputShape);
        return $outputShape;
    }

    //def backward(self, dout):
    //    dx = self.out * dout
    //    sumdx = np.sum(dx, axis=1, keepdims=True)
    //    dx -= self.out * sumdx
    //    return dx

    // MSE
    public function meanSquaredError(NDArray $trues, NDArray $predicts) : float
    {
        $la = $this->la;
        //  E = (1/N) * sum((Yk-Tk)**2)
        $N = $predicts->size();
        return $la->sum($la->square(
            $la->axpy($trues,$la->copy($predicts),-1.0))) / $N;
    }

    public function dMeanSquaredError(NDArray $trues,NDArray $predicts) : NDarray
    {
        $la = $this->la;
        // dx = 2/N * (Yk-Tk)
        return $la->scal(2/$predicts->size(),
            $la->axpy($trues,$la->copy($predicts),-1.0));
    }

    // MAE
    //def mean_absolute_error(y_true, y_pred):
    //    if not K.is_tensor(y_pred):
    //        y_pred = K.constant(y_pred)
    //    y_true = K.cast(y_true, y_pred.dtype)
    //    return K.mean(K.abs(y_pred - y_true), axis=-1)


    public function sparseCategoricalCrossEntropy(
        NDArray $trues, NDArray $predicts) : float
    {
        $la = $this->la;
        $ndim = $trues->ndim();
        $orgTrues = $trues;
        $orgPredicts = $predicts;
        if($ndim==1){
            ;
        } elseif($ndim==2) {
            if($predicts->ndim()<3){
                $msg = '['.implode(',',$orgTrues->shape()).'] ['.implode(',',$orgPredicts->shape()).']';
                throw new InvalidArgumentException('unmatch shape of dimensions:'.$msg);
            }
            $trues = $trues->reshape([$trues->size()]);
            $shape = $predicts->shape();
            $batchSize = array_shift($shape);
            $batchSize *= array_shift($shape);
            array_unshift($shape,$batchSize);
            $predicts = $predicts->reshape($shape);
        } elseif($ndim>2) {
            throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        }
        $shape = $predicts->shape();
        $batchSize = array_shift($shape);
        if($trues->shape()!=[$batchSize]){
            $msg = '['.implode(',',$orgTrues->shape()).'] ['.implode(',',$orgPredicts->shape()).']';
            throw new InvalidArgumentException('unmatch shape of dimensions:'.$msg);
        }

        //  E = - 1/N * sum-n(sum-k(t-nk * log(y-nk)))
        return -1.0 * $la->sum($la->log($la->increment(
                //$la->selectAxis1($predicts,$trues),
                $la->select($predicts,$trues,$axis=1),
                $this->epsilon))) / $batchSize;
    }

    public function dSparseCategoricalCrossEntropy(
        NDArray $trues, NDArray $predicts, bool $fromLogits=null) : NDArray
    {
        $la = $this->la;
        $origPredictsShape = $predicts->shape();
        $ndim = $trues->ndim();
        if($ndim==1){
            ;
        } elseif($ndim==2) {
            $trues = $trues->reshape([$trues->size()]);
            $predictsShape = $origPredictsShape;
            $inputDim = array_pop($predictsShape);
            $predicts = $predicts->reshape(
                [array_product($predictsShape),$inputDim]
                );
        } elseif($ndim>2) {
            throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        }
        if($trues->size()!=$predicts->shape()[0]){
            $msg = '['.implode(',',$trues->shape()).'] ['.implode(',',$predicts->shape()).']';
            throw new InvalidArgumentException('unmatch shape of dimensions:'.$msg);
        }
        $numClass = $predicts->shape()[1];
        if($fromLogits) {
            // dx = (y - t)      #  t=onehot(trues), y=softmax(x)
            $dInputs = $la->copy($predicts);
            $la->onehot($trues,$numClass,-1,$dInputs);
        } else {
            // dx = - trues / predicts
            $trues = $la->onehot($trues,$numClass);
            $dInputs = $la->scal(-1.0,$la->multiply($trues,
                $la->reciprocal($la->copy($predicts),$this->epsilon)));
        }
        return $dInputs->reshape($origPredictsShape);
    }

    public function categoricalCrossEntropy(
        NDArray $trues, NDArray $predicts) : float
    {
        $la = $this->la;
        if($trues->shape()!=$predicts->shape()){
            $msg = '['.implode(',',$trues->shape()).'] ['.implode(',',$predicts->shape()).']';
            throw new InvalidArgumentException('must be same shape of dimensions:'.$msg);
        }
        //  E = - 1/N * sum-n(sum-k(t-nk * log(y-nk)))
        $batchSize = $predicts->shape()[0];
        return -1.0 * $la->sum($la->multiply($trues,
            $la->log($la->increment($la->copy($predicts),$this->epsilon)))) / $batchSize;

        // way for clip
        //$predicts = $this->la->maximum($this->epsilon,
        //    $this->la->minimum(1-$this->epsilon,$this->la->copy($predicts)));
        //return -1.0 * $this->la->sum($this->la->multiply($trues,
        //    $this->la->log($predicts))) / $batchSize;
    }

    public function dCategoricalCrossEntropy(
        NDArray $trues, NDArray $predicts, bool $fromLogits=null) : NDArray
    {
        $la = $this->la;
        if($trues->shape()!=$predicts->shape()){
            $msg = '['.implode(',',$trues->shape()).'] ['.implode(',',$predicts->shape()).']';
            throw new InvalidArgumentException('must be same shape of dimensions:'.$msg);
        }
        if($fromLogits) {
            //  dx = y - t   :  y = softmax(x)
            $dInput = $la->axpy($trues, $la->copy($predicts), -1);
            return $dInput;
        } else {
            // dx = - trues / predicts
            return $la->scal(-1.0,$la->multiply($trues,
                $la->reciprocal($la->copy($predicts),$this->epsilon)));
        }
    }

    public function rnnGetTimestep(
        NDArray $source,int $step) : NDArray
    {
        if($source->ndim()!=3){
            throw new InvalidArgumentException('array must be 3D');
        }
        [$batch,$steps,$feature] = $source->shape();
        $values = $this->la->slice(
            $source,
            [0,$step],[-1,1]
        );

        return $values->reshape([$batch,$feature]);
    }

    public function rnnSetTimestep(
        NDArray $dest,int $step,NDArray $values) : NDArray
    {
        if($dest->ndim()!=3){
            throw new InvalidArgumentException('array must be 3D');
        }
        [$batch,$steps,$feature] = $dest->shape();
        $values = $values->reshape([$batch,1,$feature]);
        $this->la->stick(
            $values,
            $dest,
            [0,$step],[-1,1]
        );
        return $dest;
    }

    public function rnn(
        $stepFunction,
        NDArray $inputs,
        array $initialStates,
        bool $training,
        NDArray $outputs=null,
        bool $goBackwards=null
    ) : array
    {
        $inputLength = $inputs->shape()[1];
        $states_t = $initialStates;
        $calcStates = [];
        $tm = range(0,$inputLength-1);
        if($goBackwards){
            $tm = array_reverse($tm);
        }
        foreach($tm as $t){
            $calcState = new \stdClass();
            $calcStates[$t] = $calcState;
            [$outputs_t, $states_t] = $stepFunction($this->rnnGetTimestep($inputs, $t), $states_t,$training,$calcState);
            if($outputs){
                $this->rnnSetTimestep($outputs,$t,$outputs_t);
            }
        }
        if($outputs===null){
            $outputs=$outputs_t;
        }
        return [$outputs, $states_t, $calcStates];
    }

    public function rnnBackward(
        $stepFunction,
        NDArray $dOutputs,
        array $dStates,
        array $calcStates,
        NDArray $dInputs,
        bool $goBackwards=null
    ) : array
    {
        $ndim = $dOutputs->ndim();
        if($ndim == 2) {
            $seq = false;
            $zero = $this->zerosLike($dOutputs);
        } elseif($ndim == 3) {
            $seq = true;
            $zero = null;
        } else {
            throw new InvalidArgumentException('invalid dOutputs shape');
        }
        if($dInputs->ndim()!=3){
            throw new InvalidArgumentException('invalid dInputs shape');
        }
        $inputLength=$dInputs->shape()[1];
        $tm = range(0,$inputLength-1);
        if(!$goBackwards){
            $tm = array_reverse($tm);
        }
        $doutputs_t = null;
        $states_t = $dStates;
        foreach($tm as $t){
            if($seq){
                $doutputs_t = $this->rnnGetTimestep($dOutputs, $t);
            }else{
                if($doutputs_t==null){
                    $doutputs_t = $dOutputs;
                }else{
                    $doutputs_t = $zero;
                }
            }
            $calcState = $calcStates[$t];
            [$inputs_t, $states_t] = $stepFunction($doutputs_t, $states_t,$calcState);
            $this->rnnSetTimestep($dInputs,$t,$inputs_t);
        }
        return [$dInputs, $states_t];
    }
/*
    public function binaryCrossEntropy(
        NDArray $trues, NDArray $predicts, bool $fromLogits=null) : float
    {
        if($trues->shape()!=$predicts->shape()){
            throw new InvalidArgumentException('must be same shape of dimensions');
        }
        $la = $this->la;
        if(!$fromLogits) {
            //$predicts = $this->la->maximum($this->epsilon,
            //    $this->la->minimum(1-$this->epsilon,$this->la->copy($predicts)));
            // z = log( x / (1-x) )
            $predicts = $la->log($la->multiply($predicts,
                                            $la->reciprocal($predicts,1,-1)));
            // sigmoid(z) = 1 / (1 + exp(-z))
            //            = 1 / (1 + exp(-log(x/(1-x))))
            //            = x
        }
        $batchSize = $predicts->shape()[0];
        // E = t * -log(sigmoid(x)) + (1 - t) * -log(1 - sigmoid(x))
        //   = x - x * t + log(1 + exp(-x))
        return $la->sum($la->axpy($predicts,$la->axpy(
            $la->multiply($predicts,$la->copy($trues)),
            $la->log($la->increment($la->exp(
                $la->scal(-1,$la->copy($predicts)))))))) / $batchSize;
        // python
        //if not from_logits:
        //    output = np.clip(output, 1e-7, 1 - 1e-7)
        //    output = np.log(output / (1 - output))
        //output = sigmoid(output)
        //  E = -target * log(output) + -(1 - target) * log(1 - output))
        //return (-target * np.log(output)) +
        //        (-(1 - target) * np.log(1 - output))

        // part2
        //self.loss = cross_entropy_error(self.t, np.c_[1 - self.y, self.y])
        //  cross_entropy_error = - 1/N * sum-n(sum-k(t-nk * log(y-nk)))
    }

    public function dBinaryCrossEntropy(
        NDArray $trues, NDArray $predicts, bool $fromLogits=null) : NDArray
    {
        $la = $this->la;
        if($trues->shape()!=$predicts->shape()){
            throw new InvalidArgumentException('must be same shape of dimensions');
        }
        if($fromLogits) {
            // dx = y - t    :  y = sigmoid(x)
            return $la->axpy($trues,$la->copy($predicts),-1);
        } else {
            // dx = - trues / predicts
            return $la->scal(-1.0,$la->multiply($trues,
                $la->reciprocal($la->copy($predicts),$this->epsilon)));
        }
        // part2
        //dx = (self.y - self.t) * dout / batch_size
    }
*/

    public function equalTest($a,$b)
    {
        $mo = $this->matrixOperator;
        if($a instanceof NDArray) {
            if(!($b instanceof NDArray))
                throw new InvalidArgumentException('NDArrays must be of the same type.');
            if($a->shape()!=$b->shape())
                return false;
            $delta = $mo->zerosLike($b);
            $this->la->copy($b,$delta);
            $this->la->axpy($a,$delta,-1.0);
            $delta = $this->la->asum($delta);
        } elseif(is_numeric($a)) {
            if(!is_numeric($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = abs($a - $b);
        } elseif(is_bool($a)) {
            if(!is_bool($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = ($a==$b)? 0 : 1;
        } else {
            throw new InvalidArgumentException('Values must be DNArray or float or int.');
        }

        if($delta < $this->equalEpsilon) {
            return true;
        } else {
            return false;
        }
    }
}
