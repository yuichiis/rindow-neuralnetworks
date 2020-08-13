<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractLayer extends AbstractLayerBase
{
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    final public function forward(NDArray $inputs, bool $training) : NDArray
    {
        $this->assertInputShape($inputs);

        $outputs = $this->call($inputs, $training);

        $this->assertOutputShape($outputs);
        return $outputs;
    }

    final public function backward(NDArray $dOutputs) : NDArray
    {
        $this->assertOutputShape($dOutputs);

        $dInputs = $this->differentiate($dOutputs);

        $this->assertInputShape($dInputs);
        return $dInputs;
    }
}
