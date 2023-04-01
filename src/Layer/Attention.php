<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;

class Attention extends AbstractLayerBase
{
    use GenericUtils;
    use GradientUtils;
    protected $backend;
    protected $useScale;
    protected $scale;
    protected $dScale;
    protected $scoresShape;
    protected ?array $unbackpropagatables = null;

    //protected $returnAttentionScores;

    //protected $query;
    //protected $value;
    //protected $key;
    //protected $attentionWeight;

    public function __construct(
        object $backend,
        array $input_shapes=null,
        bool $use_scale=null,
        string $name=null,
    )
    {
        // defaults
        $input_shapes = $input_shapes ?? null;
        $name = $name ?? null;
        $use_scale = $use_scale ?? false;

        $this->backend = $K = $backend;
        $this->inputShape = $input_shapes;
        $this->useScale = $use_scale;
        if($this->useScale) {
            $this->scale = $K->array(1.0);
            $this->dScale = $K->array(0.0);
            $this->allocateWeights(1);
        }
        $this->initName($name,'attention');
    }

    public function build($variables=null, array $sampleWeights=null)
    {
        $K = $this->backend;
        $inputShapes = $this->normalizeInputShape($variables);
        if(count($inputShapes)!=2&&count($inputShapes)!=3) {
            throw new InvalidArgumentException('num of inputs must be 2 or 3: inputs is '.count($inputShapes));
        }
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)||count($shape)!=2) {
                if(is_array($shape)) {
                    $type = '['.implode(',',$shape).']';
                } else {
                    $type = gettype($shape);
                }
                throw new InvalidArgumentException('input_shapes must be the list of shape: '.$type.' included in #'.$idx.'.');
            }
        }
        [$tq, $dim] = $inputShapes[0];  // Query
        [$tv, $tdim] = $inputShapes[1]; // Value
        if($dim!=$tdim) {
            throw new InvalidArgumentException('Unmatch query shape and value shape:'.
            '['.implode(',',$inputShapes[0]).'],['.implode(',',$inputShapes[1]).']');
        }
        if(count($inputShapes)==3) {
            if($inputShapes[1]!=$inputShapes[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
        }
        $this->outputShape = [$tq,$dim];
        $this->scoresShape = [$tq,$tv];
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

    protected function assertInputShapes(array $inputs,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape in '.$this->name.':'.$direction);
        }
        if(count($inputs)!=2 && count($inputs)!=3) {
            throw new InvalidArgumentException('Must have 2 or 3 arguments in '.$this->name.':'.$direction);
        }
        $tq = $this->inputShape[0][0];
        $dim = $this->inputShape[0][1];
        $tv = $this->inputShape[1][0];
        $qshape = $inputs[0]->shape();
        if($qshape[1]!=$tq||$qshape[2]!=$dim){
            throw new InvalidArgumentException('Unmatch query shape [b,'.$tq.','.$dim.'] NDArray.'.
                ' ['.implode(',',$qshape).'] given in '.$this->name.':'.$direction);
        }
        if($qshape[1]!=$tq||$qshape[2]!=$dim){
            throw new InvalidArgumentException('Unmatch query shape [b,'.$tq.','.$dim.'] NDArray.'.
                ' ['.implode(',',$qshape).'] given in '.$this->name.':'.$direction);
        }
        $vshape = $inputs[0]->shape();
        if($vshape[1]!=$tq||$vshape[2]!=$dim){
            throw new InvalidArgumentException('Unmatch value shape [b,'.$tq.','.$dim.'] NDArray.'.
                ' ['.implode(',',$vshape).'] given in '.$this->name.':'.$direction);
        }
        if($qshape[0]!=$vshape[0]) {
            throw new InvalidArgumentException('Unmatch batch size.:'.
                ' ['.implode(',',$qshape).'],['.implode(',',$vshape).'] given in '.$this->name.':'.$direction);
        }
        if(count($inputs)==3) {
            $kshape = $inputs[0]->shape();
            if($kshape!=$vshape){
                throw new InvalidArgumentException('Unmatch value shape and key shape.:'.
                    ' ['.implode(',',$vshape).'],['.implode(',',$kshape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    protected function assertScoresShape(NDArray $scores,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->scoresShape===null) {
            throw new InvalidArgumentException('Uninitialized scores shape');
        }
        $shape = $scores->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->scoresShape) {
            $shape = $this->shapeToString($shape);
            $scoresShape = $this->shapeToString($this->scoresShape);
            throw new InvalidArgumentException('unmatch scores shape: '.$shape.', must be '.scoresShape.' in '.$this->name.':'.$direction);
        }
    }

    public function backward(array $dOutputs,ArrayAccess $grads=null,array $oidsToCollect=null) : array
    {
        if(count($dOutputs)!=1&&count($dOutputs)!=2) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $dOutputs = $dOutputs[0];
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShapes($dInputs,'backward');
        $this->collectGradients($this->backend,array_map(null,$this->trainableVariables(),$this->getGrads()),
            $grads,$oidsToCollect);
        return $dInputs;
    }

    protected function call(
        array $inputs,
        bool $training=null,
        bool $returnAttentionScores=null,
        array $mask=null,
        )
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
        
        if($this->useScale) {
            // scores = scores / sqrt(qk) 
            $scale = $K->scalar($this->scale);
            $scores = $K->update_scale($scores,$scale);
            $container->scale = $scale;
            $container->scores = $K->copy($scores);
        }
        $queryMask = null;
        $valueMask = null;
        if($mask) {
            [$queryMask,$valueMask] = $mask;
        }
        if($valueMask) {
            // scores = [batch_size, Tq, Tv] => [Tq, batch_size, Tv]
            $scoresShape = $scores->shape();
            $origScoreShape = $scoresShape;
            $Tv = array_pop($scoresShape);
            $Tq = array_pop($scoresShape);
            $scores = $scores->reshape([(int)array_product($scoresShape),$Tq,$Tv]);
            $scores = $K->transpose($scores,perm:[1,0,2]);
            $scores = $scores->reshape(array_merge([$Tq],$scoresShape,[$Tv]));
            // broadcast mask
            // scores = [Tq, batch_size, Tv]
            // valueMask = [batch_size, Tv] 
            $valueMask = $K->cast($K->equal($valueMask,$K->zerosLike($valueMask)),$scores->dtype());
            $valueMask = $K->scale(-1e9,$valueMask);
            $K->update_add($scores,$valueMask);
            // scores = [Tq, batch_size, Tv] => [batch_size, Tq, Tv]
            $scores = $scores->reshape([$Tq,(int)array_product($scoresShape),$Tv]);
            $scores = $K->transpose($scores,perm:[1,0,2]);
            $scores = $scores->reshape($origScoreShape);
        }
        // weights = softmax(scores)
        $attentionWeight = $K->softmax($scores);

        // vector = weights * value
        // scores = [batch_size, Tq, Tv]
        // value  = [batch_size, Tv, dim]
        // vector = [batch_size, Tq, dim]
        $contextVector = $K->matmul($attentionWeight, $value);

        if($queryMask) {
            // queryMask = [batch_size, Tq]
            // vector = [batch_size, Tq, dim] => [dim, batch_size, Tq]
            $queryMask = $K->cast($queryMask,$contextVector->dtype());
            $queryMask = $queryMask->reshape([(int)array_product($queryMask->shape())]);
            [$batchSize,$Tq,$dim] = $contextVector->shape();
            $contextVector = $contextVector->reshape([$batchSize*$Tq, $dim]);
            $contextVector = $K->update_mul($contextVector, $queryMask, trans:true);
            $contextVector = $contextVector->reshape([$batchSize, $Tq, $dim]);
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
            // queryMask = [batch_size*Tq]
            // vector = [batch_size, Tq, dim] => [dim, batch_size, Tq]
            [$batchSize,$Tq,$dim] = $dOutputs->shape();
            $dOutputs = $K->copy($dOutputs->reshape([$batchSize*$Tq, $dim]));
            $dOutputs = $K->update_mul($dOutputs, $container->queryMask, trans:true);
            $dOutputs = $dOutputs->reshape([$batchSize, $Tq, $dim]);
        }

        $dAttentionWeight = $K->matmul($dOutputs,$container->value,$transA=false,$transB=true);
        $dValue = $K->matmul($container->attentionWeight,$dOutputs,$transA=true,$transB=false);

        $dScores = $K->dSoftmax($dAttentionWeight,$container->attentionWeight);

        // valueMask is dAdd so it is passed through.

        if(isset($container->scale)) {
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

    protected function numOfOutputs(?array $options)
    {
        if($options['returnAttentionScores'] ?? false) {
            return 2;
        }
        return 1;
    }

    final public function __invoke(...$args)
    {
        return $this->forward(...$args);
    }

    /**
    *  @param array<Variable>  $inputs
    *  @return array<Variable>|Variable
    */
    public function forward(
        array $inputs, 
        Variable|bool $training=null, 
        Variable|bool $returnAttentionScores=null,
        array $mask=null,
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
        $options['training'] = $training;
        $options['returnAttentionScores'] = $returnAttentionScores;
        $rawMask = null;
        if($mask) {
            if(count($mask)!=2) {
                throw new InvalidArgumentException('mask must be list of the two of masks as queryMask and valueMask');
            }
            [$mask,$rawMask] = $this->packAndUnpackVariables($this->backend,$mask,unbackpropagatable:true);
            $options['queryMask'] = $mask[0];
            $options['valueMask'] = $mask[1];
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

    /**
     * Call from SessionFunc in compiled graph
     */
    public function _rawCall(array $inputs,array $options)
    {
        $training = $options['training'] ?? null;
        $queryMask = $options['queryMask'] ?? null;
        $valueMask = $options['valueMask'] ?? null;
        $mask = null;
        if($queryMask) {
            $mask = [$queryMask,$valueMask];
        }
        $returnAttentionScores = $options['returnAttentionScores'] ?? null;
        $outputs = $this->call(
            $inputs,
            training:$training,
            returnAttentionScores:$returnAttentionScores,
            mask:$mask,
        );
        if(!is_array($outputs)) {
            $outputs = [$outputs];
        }
        return $outputs;
    }

}
