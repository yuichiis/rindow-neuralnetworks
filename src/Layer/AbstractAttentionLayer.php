<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;


/**
 *
 */
abstract class AbstractAttentionLayer extends AbstractLayerBase
{
    use GenericUtils;
    use GradientUtils;

    abstract protected function call(
        array $inputs,
        bool $training=null,
        bool $returnAttentionScores=null,
        array $masks=null,
    ) : NDArray|array;

    /**
     * @return array<NDArray>
     */
    abstract protected function differentiate(NDArray $dOutputs) : array;

    /**
     * @param array<NDArray> $inputs
     */
    protected function assertInputShapes(array $inputs,string $direction) : void
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape in '.$this->name.':'.$direction);
        }
        if(count($inputs)!=2 && count($inputs)!=3) {
            throw new InvalidArgumentException('Must have 2 or 3 arguments in '.$this->name.':'.$direction);
        }
        //$tq = $this->inputShape[0][0];
        //$dim = $this->inputShape[0][1];
        //$tv = $this->inputShape[1][0];
        $qshape = $inputs[0]->shape();
        $batchNum = array_shift($qshape);
        $vshape = $inputs[1]->shape();
        $vbatchNum = array_shift($vshape);
        if($batchNum!=$vbatchNum) {
            throw new InvalidArgumentException('Unmatch batch size of query and value: '.
                "query=[$batchNum,".implode(',',$qshape)."],".
                "value=[$vbatchNum,".implode(',',$vshape)."]".
                "in ".$this->name.':'.$direction);
        }
        if($this->inputShape[0]!=$qshape){
            throw new InvalidArgumentException('Unmatch query shape '.
                ' [b,'.implode(',',$this->inputShape[0]).'] NDArray.'.
                ' ['.$batchNum.','.implode(',',$qshape).'] given in '.$this->name.':'.$direction);
        }
        if($this->inputShape[1]!=$vshape){
            throw new InvalidArgumentException('Unmatch value shape '.
                ' [b,'.implode(',',$this->inputShape[1]).'] NDArray.'.
                ' ['.$vbatchNum.','.implode(',',$vshape).'] given in '.$this->name.':'.$direction);
        }
        if(count($inputs)==3) {
            $kshape = $inputs[2]->shape();
            $kbatchNum = array_shift($kshape);
            if($vbatchNum!=$kbatchNum) {
                throw new InvalidArgumentException('Unmatch batch size of value and key: '.
                    "query=[$vbatchNum,".implode(',',$vshape)."],".
                    "value=[$kbatchNum,".implode(',',$kshape)."]".
                    "in ".$this->name.':'.$direction);
            }
            if($kshape!=$vshape){
                throw new InvalidArgumentException('Unmatch value shape and key shape.:'.
                    ' ['.implode(',',$vshape).'],['.implode(',',$kshape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    protected function assertScoresShape(NDArray $scores,string $direction) : void
    {
        if(!$this->shapeInspection) {
            return;
        }
        if($this->scoresShape===null) {
            throw new InvalidArgumentException('Uninitialized scores shape');
        }
        $shape = $scores->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->scoresShape) {
            $shape = $this->shapeToString($shape);
            $scoresShape = $this->shapeToString($this->scoresShape);
            throw new InvalidArgumentException('unmatch scores shape: '.$shape.', must be '.$scoresShape.' in '.$this->name.':'.$direction);
        }
    }

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<NDArray> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(
        array $dOutputs,
        ArrayAccess $grads=null,
        array $oidsToCollect=null
        ) : array
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

    /**
     * @param array<string,mixed> $options
     */
    protected function numOfOutputs(?array $options) : int
    {
        if($options['returnAttentionScores'] ?? false) {
            return 2;
        }
        return 1;
    }

    /**
     * @return array<Variable>|Variable
     */
    final public function __invoke(mixed ...$args) : array|NDArray
    {
        return $this->forward(...$args);
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
            );
            $rawOutputs = $this->makeMultiMaskedValues($rawInputs, $rawOutputs);
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
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array
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
            masks:$mask,
        );
        if(!is_array($outputs)) {
            $outputs = [$outputs];
        }
        $values = $this->makeMultiMaskedValues($inputs, $outputs);
        return $values;
    }
}
