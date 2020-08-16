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
        $this->assertInputShape($inputs,'forward');
        $this->assertStatesShape($initialStates,'forward');
        $results = $this->call($inputs,$training,$initialStates,$options);
        if(is_array($results)) {
            [$outputs,$states] = $results;
            $this->assertStatesShape($states,'forward');
        } elseif($results instanceof NDArray) {
            $outputs = $results;
        }
        $this->assertOutputShape($outputs,'forward');
        return $results;
    }

    final public function backward(NDArray $dOutputs, array $dStates=null)
    {
        $this->assertOutputShape($dOutputs,'backward');
        $this->assertStatesShape($dStates,'backward');

        $results = $this->differentiate($dOutputs,$dStates);

        if(is_array($results)) {
            [$dInputs,$dStates] = $results;
            $this->assertStatesShape($dStates,'backward');
        } elseif($results instanceof NDArray) {
            $dInputs = $results;
        }
        $this->assertInputShape($dInputs,'backward');
        return $results;
    }
}
