<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface RNNCell extends LayerBase
{
    public function forward(NDArray $inputs, array $states, bool $training,array $options=null) : array;
    public function backward(NDArray $dOutputs,array $dStates) : NDArray;
}