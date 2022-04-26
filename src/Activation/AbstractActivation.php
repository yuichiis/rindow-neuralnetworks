<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractActivation implements Activation
{
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    protected $states;
    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function setStates($states) : void
    {
        $this->states = $states;
    }

    public function forward(NDArray $inputs, bool $training) : NDArray
    {
        if($this->states===null) {
            $this->states = new \stdClass();
        }
        $outputs = $this->call($inputs,$training);
        return $outputs;
    }

    public function backward(NDArray $dOutputs) : NDArray
    {
        try {
            $dInputs = $this->differentiate($dOutputs);
        } finally {
            $this->states = null;
        }
        return $dInputs;
    }
}
