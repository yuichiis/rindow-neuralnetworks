<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

class Tanh extends AbstractActivation
{
    protected $mask;

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $outputs = $K->tanh($inputs);
        $this->outputs = $outputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        // dx = dy * (1 - y^2)
        $dInputs = $K->mul($dOutputs,$K->increment($K->scale(-1,$K->square($this->outputs)),1));
        return $dInputs;
    }
}
