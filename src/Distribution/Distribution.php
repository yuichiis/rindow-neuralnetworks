<?php
namespace Rindow\NeuralNetworks\Distribution;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;

interface Distribution
{
    public function logProb(Variable $value) : Variable;

    public function entropy() : Variable;

    public function sample() : Variable;
}
