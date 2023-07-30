<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use ArrayAccess;

/**
 *
 */
interface Metric
{
    public function __invoke(...$args) : mixed;
    public function name() : string;
    public function forward(NDArray $true, NDArray $predicts) : float;
}
