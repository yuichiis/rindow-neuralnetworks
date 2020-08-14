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

    final public function forward(NDArray $inputs, bool $training, array $initialStates=null,array $options=null)
    {
        $this->assertInputShape($inputs);

        $results = $this->call($inputs,$training,$initialStates,$options);
        if(is_array($results)) {
            [$outputs,$states] = $results;
        } elseif($results instanceof NDArray) {
            $outputs = $results;
        }
        $this->assertOutputShape($outputs);
        return $results;
    }

    final public function backward(NDArray $dOutputs, array $dStates=null)
    {
        $this->assertOutputShape($dOutputs);

        $results = $this->differentiate($dOutputs,$dStates);

        if(is_array($results)) {
            [$dInputs,$dStates] = $results;
        } elseif($results instanceof NDArray) {
            $dInputs = $results;
        }
        $this->assertInputShape($dInputs);
        return $results;
    }
}
