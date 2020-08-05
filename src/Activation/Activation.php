<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

interface Activation
{
    protected function call(NDArray $inputs, bool $training) : NDArray;
    protected function differentiate(NDArray $dOutputs) : NDArray;
}
