<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class StopGradient extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $outputs = [];
        $unbackpropagatables = [];
        foreach($inputs as $val) {
            $outputs[] = $K->copy($val);
            $unbackpropagatables[] = true;
        }
        $this->unbackpropagatables = $unbackpropagatables;
        return $outputs;
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $dInputs = [];
        foreach($dOutputs as $val) {
            $dInputs[] = $K->zerosLike($val);
        }
        return $dInputs;
    }
}
