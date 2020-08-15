<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractRNNLayer extends AbstractLayerBase implements RNNLayer
{
    abstract protected function call(NDArray $inputs, bool $training, array $initialStates=null, array $options=null);
    abstract protected function differentiate(NDArray $dOutputs, array $dStates=null);

    public function setShapeInspection(bool $enable)
    {
        parent::setShapeInspection($enable);
        $this->cell->setShapeInspection($enable);
    }

    final public function forward(NDArray $inputs, bool $training, array $initialStates=null,array $options=null)
    {
        $this->assertInputShape($inputs);
        $this->assertStatesShape($initialStates);
        $results = $this->call($inputs,$training,$initialStates,$options);
        if(is_array($results)) {
            [$outputs,$states] = $results;
            if(!is_array($states))
                throw new \Exception('abstractrnn');
            $this->assertStatesShape($states);
        } elseif($results instanceof NDArray) {
            $outputs = $results;
        }
        $this->assertOutputShape($outputs);
        return $results;
    }

    final public function backward(NDArray $dOutputs, array $dStates=null)
    {
        $this->assertOutputShape($dOutputs);
        $this->assertStatesShape($dStates);

        $results = $this->differentiate($dOutputs,$dStates);

        if(is_array($results)) {
            [$dInputs,$dStates] = $results;
            $this->assertStatesShape($dStates);
        } elseif($results instanceof NDArray) {
            $dInputs = $results;
        }
        $this->assertInputShape($dInputs);
        return $results;
    }
}
