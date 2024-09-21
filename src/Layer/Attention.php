<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class Attention extends AbstractAttentionLayer
{
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
     */
    public function __construct(
        object $backend,
        array $input_shapes=null,
        bool $use_scale=null,
        bool $do_not_expand_mask=null,
        string $name=null,
    )
    {
        // defaults
        $input_shapes = $input_shapes ?? null;
        $name = $name ?? null;
        $use_scale = $use_scale ?? false;
        $do_not_expand_mask = $do_not_expand_mask ?? false;

        parent::__construct($backend);
        $K = $backend;
        $this->inputShape = $input_shapes;
        $this->useScale = $use_scale;
        $this->doNotExpandMask = $do_not_expand_mask;
        if($this->useScale) {
            $this->scale = $K->array(1.0);
            $this->dScale = $K->array(0.0);
            $this->allocateWeights(1);
        }
        $this->initName($name,'attention');
    }

    public function build(mixed $variables=null, array $sampleWeights=null) : void
    {
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
        $query = $inputShapes[0];  // Query
        $dim = array_pop($query);
        $tq  = array_pop($query);
        $value = $inputShapes[1]; // Value
        $tdim = array_pop($value);
        $tv =   array_pop($value);
        if($dim!=$tdim || $query!=$value) {
            throw new InvalidArgumentException('Unmatch query shape and value shape:'.
            '['.implode(',',$inputShapes[0]).'],['.implode(',',$inputShapes[1]).']');
        }
        if(count($inputShapes)==3) {
            if($inputShapes[1]!=$inputShapes[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
        }
        $this->outputShape = array_merge($query,[$tq,$dim]);
        $this->scoresShape = array_merge($query,[$tq,$tv]);
        $this->syncWeightVariables();
    }

    public function getParams() : array
    {
        if($this->useScale) {
            return [$this->scale];
        } else {
            return [];
        }
    }

    public function getGrads() : array
    {
        if($this->useScale) {
            return [$this->dScale];
        } else {
            return [];
        }
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'input_shapes'=>$this->inputShape,
            ],
        ];
    }

    protected function expandMask(NDArray $sourceMask,NDArray $target) : NDArray
    {
        $K = $this->backend;
        $mask = $sourceMask;
        $maskShape = $mask->shape();
        $targetShape = $target->shape();
        foreach (array_map(null,$maskShape,$targetShape) as $axis => [$mT,$T]) {
            if($mT==1 && $T!=1) {
                $mask = $K->repeat($mask,$T,axis:$axis,keepdims:true);
            } elseif($mT!=$T) {
                throw new InvalidArgumentException('Incompatible shapes for broadcasting: '.
                    '['.implode(',',$sourceMask->shape()).'] vs. ['.implode(',',$target->shape()).']');
            }
        }
        return $mask;
    }

    /**
     * @param array<NDArray> $inputs
     * @param array{NDArray,NDArray} $mask
     * @return NDArray|array<NDArray>
     */
    protected function call(
        array $inputs,
        bool $training=null,
        bool $returnAttentionScores=null,
        array $masks=null,
        ) : NDArray|array
    {
        $K = $this->backend;
        $container = $this->container();
        $query = $inputs[0];
        $value = $inputs[1];
        if(count($inputs)==3) {
            $key = $inputs[2];
            $container->sameKey = false;
        } else {
            $key = $inputs[1];
            $container->sameKey = true;
        }
        // scores = query * key
        //
        // query  = [batch_size, Tq, dim]
        // key    = [batch_size, Tv, dim]
        // scores = [batch_size, Tq, Tv]
        $scores = $K->matmul($query, $key, null, $tranB=true);
        
        $container->useScale = $this->useScale;
        if($this->useScale) {
            // scores = scores / sqrt(qk) 
            $scale = $K->scalar($this->scale);
            $scores = $K->update_scale($scores,$scale);
            $container->scale = $scale;
            $container->scores = $K->copy($scores);
        }
        $queryMask = null;
        $valueMask = null;
        if($masks) {
            [$queryMask,$valueMask] = $masks;
        }
        if($valueMask) {
            if($valueMask->dtype()==NDArray::bool || $K->isInt($valueMask)) {
                $valueMask = $K->cast($valueMask,$scores->dtype());
            }
            $valueMask = $K->less($valueMask,0.5);              // (mask<0.5)>1.0 , (mask>0.5)=>0.0
            // scores += (-1e9*valueMask)
            if(!$this->doNotExpandMask) { // Broadcasting 
                // scores = [batch_size, Tq, Tv]
                // valueMask = [batch_size, Tv]
                $scoresShape = $scores->shape();
                $Tv = array_pop($scoresShape);
                $Tq = array_pop($scoresShape);
                $maskShape = $valueMask->shape();
                $mTv = array_pop($maskShape);
                if($maskShape!=$scoresShape||$Tv!=$mTv) {
                    throw new InvalidArgumentException('unmatch inputs and queryMask.'.
                    ' scores:['.implode(',',$scores->shape()).']'.
                    ' given mask:['.implode(',',$valueMask->shape()).']');
                }
                // scores = [batch_size, Tq, Tv]
                // valueMask = [batch_size, Tv] =repeat=> [batch_size, Tq, Tv]
                $valueMask = $K->repeat($valueMask,$Tq,axis:-1);
            } else { // No Broadcasting 
                // scores += (-1e9*valueMask)
                $valueMask = $this->expandMask($valueMask,$scores);
            }
            $K->update_add($scores,$valueMask,alpha:-1e9);
        }
        // weights = softmax(scores)
        $attentionWeight = $K->softmax($scores);

        // vector = weights * value
        // scores = [batch_size, Tq, Tv]
        // value  = [batch_size, Tv, dim]
        // vector = [batch_size, Tq, dim]
        $contextVector = $K->matmul($attentionWeight, $value);

        if($queryMask) {
            if($K->isFloat($queryMask)) {
                $queryMask = $K->greater($queryMask,0.5);
            } else {
                $queryMask = $K->cast($queryMask,$contextVector->dtype());
            }
            if(!$this->doNotExpandMask) { // Broadcasting 
                // queryMask = [batch_size, Tq]
                // vector = [batch_size, Tq, dim] => [dim, batch_size, Tq]
                $shape = $contextVector->shape();
                $orgShape = $shape;
                $dim = array_pop($shape);
                if($queryMask->shape()!=$shape) {
                    throw new InvalidArgumentException('unmatch inputs and queryMask.'.
                    ' contextVector:['.implode(',',$contextVector->shape()).']'.
                    ' given mask:['.implode(',',$queryMask->shape()).']');
                }
                $Tq = array_pop($shape);
                $batchSize = (int)array_product($shape);
                $queryMask = $queryMask->reshape([(int)array_product($queryMask->shape())]);
                $contextVector = $contextVector->reshape([$batchSize*$Tq, $dim]);
                $contextVector = $K->update_mul($contextVector, $queryMask, trans:true);
                $contextVector = $contextVector->reshape($orgShape);
            } else { // No Broadcasting 
                $queryMask = $this->expandMask($queryMask,$contextVector);
                $contextVector = $K->update_mul($contextVector, $queryMask);
            }
            $container->queryMask = $queryMask;
        }

        $container->value = $value;
        $container->attentionWeight = $attentionWeight;
        $container->query = $query;
        $container->key = $key;
        if($returnAttentionScores) {
            $this->unbackpropagatables = [false,true];
            return [$contextVector,$attentionWeight];
        } else {
            return $contextVector;
        }
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        // forward:
        //   vector = weights (*) value
        // backward:
        //   dWeights = dVector (*) value^T
        //   dValue   = weights^T (*) dVector

        if(isset($container->queryMask)) {
            if(!$this->doNotExpandMask) { // Broadcasting 
                // queryMask = [batch_size*Tq]
                // vector = [batch_size, Tq, dim] => [dim, batch_size, Tq]
                $batchShape = $dOutputs->shape();
                $dim = array_pop($batchShape);
                $Tq = array_pop($batchShape);
                $batchSize = (int)array_product($batchShape);
                $dOutputs = $K->copy($dOutputs->reshape([$batchSize*$Tq, $dim]));
                $dOutputs = $K->update_mul($dOutputs, $container->queryMask, trans:true);
                $dOutputs = $dOutputs->reshape([$batchSize, $Tq, $dim]);
            } else {
                $dOutputs = $K->copy($dOutputs);
                $dOutputs = $K->update_mul($dOutputs, $container->queryMask);
            }
        }

        $dAttentionWeight = $K->matmul($dOutputs,$container->value,$transA=false,$transB=true);
        $dValue = $K->matmul($container->attentionWeight,$dOutputs,$transA=true,$transB=false);

        $dScores = $K->dSoftmax($dAttentionWeight,$container->attentionWeight);

        // valueMask is dAdd so it is passed through.

        if($container->useScale) {
            // dScale  = sum(dScales * scales)
            // dScores = dScores * scale 
            $dScale = $K->sum($K->mul($dScores,$container->scores));
            if(is_scalar($dScale)) {
                $dScale = $K->array($dScale);
            }
            $K->copy($dScale,$this->dScale);
            $K->update_scale($dScores,$container->scale);
        }

        $dQuery = $K->matmul($dScores,$container->key,$transA=false,$transB=false);
        $dKey = $K->matmul($dScores,$container->query,$transA=true,$transB=false);

        if($container->sameKey) {
            $K->update_add($dValue,$dKey);
            return [$dQuery,$dValue];
        } else {
            return [$dQuery,$dValue,$dKey];
        }
    }

}
