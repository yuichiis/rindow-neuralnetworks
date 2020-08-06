<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

class Softmax extends AbstractActivation
{
    protected $outputs;

    public function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->outputs = $K->softmax($inputs);
        return $this->outputs;
    }

    public function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        return $K->dSoftmax($dOutputs, $this->outputs);
    }
}
