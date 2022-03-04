<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Model\BuildContext;

abstract class AbstractMultiInputLayer extends AbstractLayerBase
{
    use GradientUtils;
    abstract protected function call(array $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : array;

    protected function assertInputShapes(array $inputs,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape');
        }
        if(count($inputs)!=count($this->inputShape)){
            throw new InvalidArgumentException('Unmatch num of input. inputs need '.count($this->inputShapes).' NDArray. '.count($inputs).'given in '.$this->name.':'.$direction);
        }
        $batchSize = null;
        foreach($inputs as $idx=>$input){;
            $inputShape = $this->inputShape[$idx];
            $shape = $input->shape();
            if($batchSize === null) {
                $batchSize = array_shift($shape);
            } else {
                if($batchSize != array_shift($shape)) {
                    throw new InvalidArgumentException('unmatch batch size of input '.$idx.': ['.$batchSize.'] in '.$this->name.':'.$direction);
                }
            }
            if($shape!=$inputShape) {
                $shape = $shape ? implode(',',$shape) : '';
                throw new InvalidArgumentException('unmatch shape of input '.$idx.': ['.$shape.'], must be ['.implode(',',$inputShape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    public function forward(array $inputs, bool $training)
    {
        if(BuildContext::$build) {
            return $this->build($inputs);
        }
        if(count($inputs)<2) {
            throw new InvalidArgumentException('Must have arguments greater than 2 or equal');
        }
        $this->assertInputShapes($inputs,'forward');
        $outputs = $this->call($inputs,$training);
        $this->assertOutputShape($outputs,'forward');
        return $outputs;
    }

    public function backward(array $dOutputs, ArrayAccess $grads=null,array $oidsToCollect=null) : array
    {
        if(count($dOutputs)!=1) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $dOutputs = $dOutputs[0];
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShapes($dInputs,'backward');
        $this->collectGradients($this->backend,array_map(null,$this->weights(),$this->getGrads()),
            $grads,$oidsToCollect);
        return $dInputs;
    }


    /**
    *  @param array<Variable>  $inputs
    *  @return array<Variable>
    */
    public function __invoke($inputs, bool $training)
    {
        if(!is_array($inputs)) {
            throw new InvalidArgumentException('inputs must be list of Variable');
        }
        if($this->outputShape==null) {
            $this->build($inputs);
        }
        $session = $this->preGradientProcessOnSession($inputs);
        $session->begin();
        try {
            $rawInputs = array_map(function($value){return $value->value();},$inputs);
            $outputs = $this->forward($rawInputs,$training);
        } catch(Throwable $e) {
            $session->end();
            throw $e;
        }
        $session->end();
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session, $inputs, [$outputs]);
        return $outputs[0];
    }
}
