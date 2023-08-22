<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;

class GenericMetric extends AbstractMetric
{
    protected string $name = 'generic_metric';
    protected $func;

    public function __construct(
        object $backend,
        callable $func,
        )
    {
        parent::__construct($backend);
        $this->func = $func;
    }

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        return ($this->func)($trues, $predicts);
    }
}
